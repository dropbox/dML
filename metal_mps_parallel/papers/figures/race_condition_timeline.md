# AGX Driver Race Condition Timeline

## Figure 4: Race Condition Sequence Diagram

```
TIME ───────────────────────────────────────────────────────────────────────────►

THREAD 0                              THREAD 1                    CONTEXT STATE
═════════                             ═════════                   ══════════════

    │                                     │
    │ CreateContext(slot=0)               │                       slot[0] = NULL
    │─────────────────────────►           │
    │                                     │
    │ context_0 = new Context()           │                       slot[0] = VALID ✓
    │ slot[0] = context_0                 │
    │                                     │
    │ BeginEncoding()                     │                       Thread 0: ENCODING
    │─────────────────────────►           │
    │                                     │
    │    ╔══════════════════════════════════════════════════════════════════╗
    │    ║              THE RACE WINDOW OPENS                                ║
    │    ╚══════════════════════════════════════════════════════════════════╝
    │                                     │
    │ [Thread 0 is encoding]              │ DestroyContext(slot=0)
    │ [Using context_0]                   │──────────────────────►
    │                                     │
    │                                     │ // BUG: No ownership check!
    │                                     │ delete slot[0]
    │                                     │ slot[0] = NULL         slot[0] = INVALID ✗
    │                                     │
    │    ╔══════════════════════════════════════════════════════════════════╗
    │    ║              RACE CONDITION TRIGGERED                             ║
    │    ╚══════════════════════════════════════════════════════════════════╝
    │                                     │
    │ UseContext(context_0)               │
    │─────────────────────────►           │
    │                                     │
    │ // context_0 was freed!             │
    │ // Now points to NULL/garbage       │
    │                                     │                       context_0 = NULL
    │ ldr x0, [x20, #0x5c8]               │
    │ // x20 = NULL                       │
    │                                     │
    │ ╔═════════════════════════════════════════════════════════════════════╗
    │ ║ *** CRASH ***                                                        ║
    │ ║ SIGSEGV: NULL pointer dereference at 0x5c8                          ║
    │ ║ Exception Type: EXC_BAD_ACCESS (SIGSEGV)                            ║
    │ ║ Exception Codes: KERN_INVALID_ADDRESS at 0x00000000000005c8         ║
    │ ╚═════════════════════════════════════════════════════════════════════╝
    ▼                                     ▼

```

## Figure 5: Detailed State Machine (TLA+/Lean4 Model)

```
                    ┌────────────────────────────────────────┐
                    │           INITIAL STATE                │
                    │  contexts: [NULL, NULL, NULL, NULL]    │
                    │  threads: [idle, idle]                 │
                    │  null_derefs: 0                        │
                    └──────────────────┬─────────────────────┘
                                       │
                           Thread 0: CreateContext(0)
                                       │
                                       ▼
                    ┌────────────────────────────────────────┐
                    │            STATE 1                     │
                    │  contexts: [CREATING, NULL, ...]       │
                    │  threads: [creating, idle]             │
                    └──────────────────┬─────────────────────┘
                                       │
                           Thread 0: FinishCreate(0)
                                       │
                                       ▼
                    ┌────────────────────────────────────────┐
                    │            STATE 2                     │
                    │  contexts: [VALID ✓, NULL, ...]        │
                    │  threads: [encoding, idle]             │
                    │  thread_context[0] = ctx_0             │
                    └──────────────────┬─────────────────────┘
                                       │
                   ┌───────────────────┼───────────────────┐
                   │                   │                   │
                   │    Thread 1: DestroyContext(0)        │
                   │    // THE BUG: No ownership check!    │
                   │                   │                   │
                   ▼                   │                   ▼
   ┌───────────────────────────┐       │    ┌───────────────────────────┐
   │        STATE 3 (BAD)      │       │    │      STATE 3 (GOOD)       │
   │ contexts: [INVALID ✗,...]│       │    │ contexts: [VALID ✓, ...]  │
   │ threads: [encoding, idle] │       │    │ Thread 1: BLOCKED         │
   │ !! ctx_0 DESTROYED !!     │       │    │ (mutex held by Thread 0)  │
   └───────────┬───────────────┘       │    └───────────────────────────┘
               │                       │
   Thread 0: UseContext()              │
               │                       │
               ▼                       │
   ┌───────────────────────────┐       │
   │        STATE 4 (CRASH)    │       │
   │ contexts: [NULL, ...]     │       │
   │ null_derefs: 1            │       │
   │ race_witnessed: TRUE      │       │
   │ *** SIGSEGV ***           │       │
   └───────────────────────────┘       │
                                       │
               WITH MUTEX FIX          │
                     │                 │
                     ▼                 │
   ┌───────────────────────────────────┘
   │
   ▼
   ┌────────────────────────────────────────┐
   │        STATE 3-6 (SAFE PATH)           │
   │ Step 3: Thread 1 BLOCKED by mutex      │
   │ Step 4: Thread 0 continues encoding    │
   │ Step 5: Thread 0 releases mutex        │
   │ Step 6: Thread 1 proceeds (safe)       │
   │                                        │
   │ null_derefs: 0  ✓                      │
   │ race_witnessed: FALSE  ✓               │
   └────────────────────────────────────────┘
```

## Figure 6: Mutex Protection Timeline (Fixed)

```
TIME ───────────────────────────────────────────────────────────────────────────►

                    MUTEX STATE: [FREE]

THREAD 0                              THREAD 1
═════════                             ═════════

    │ AcquireMutex()                      │
    │═════════════════►                   │
    │ SUCCESS                             │                    MUTEX: [HELD by T0]
    │                                     │
    │ CreateContext(slot=0)               │
    │──────────────────►                  │
    │                                     │
    │ BeginEncoding()                     │ AcquireMutex()
    │──────────────────►                  │═════════════════►
    │                                     │ ┌──────────────┐
    │ [Encoding safely]                   │ │   BLOCKED    │
    │                                     │ │   WAITING    │
    │ UseContext()                        │ │   for mutex  │
    │──────────────────►                  │ │              │
    │ SUCCESS (context valid!)            │ │              │
    │                                     │ │              │
    │ FinishEncoding()                    │ │              │
    │──────────────────►                  │ │              │
    │                                     │ └──────────────┘
    │ ReleaseMutex()                      │
    │═════════════════►                   │                    MUTEX: [FREE]
    │                                     │
    │                                     │ SUCCESS (acquired)  MUTEX: [HELD by T1]
    │                                     │─────────────────►
    │                                     │
    │                                     │ // NOW safe to destroy
    │                                     │ DestroyContext(slot=0)
    │                                     │──────────────────►
    │                                     │
    │                                     │ ReleaseMutex()
    │                                     │═════════════════►   MUTEX: [FREE]
    ▼                                     ▼

    NO CRASHES - Thread 0 completes before Thread 1 can destroy context
```
