#!/usr/bin/env python3
"""
lldb_agx_debug.py - LLDB Python script for debugging AGX driver race conditions

Worker: N=1480
Purpose: Phase 3.2 dynamic analysis with LLDB

Usage:
    # Method 1: Load into LLDB session
    lldb python3
    (lldb) command script import /path/to/lldb_agx_debug.py
    (lldb) agx_setup
    (lldb) run your_script.py

    # Method 2: Launch with script
    lldb -o "command script import lldb_agx_debug.py" -o "agx_setup" -- python3 test.py

    # Method 3: Attach to running process
    lldb -p <PID>
    (lldb) command script import lldb_agx_debug.py
    (lldb) agx_setup

Commands added:
    agx_setup        - Set up all breakpoints for AGX race debugging
    agx_crash_info   - Print context/encoder info at crash site
    agx_encoders     - List active encoder/context addresses
    agx_mutex_stats  - Show mutex acquisition counts (if agx_fix loaded)
"""

import lldb
import os

# Global state for tracking encoders
encoder_contexts = {}  # encoder_addr -> context_info
mutex_acquisitions = 0
mutex_contentions = 0


def __lldb_init_module(debugger, internal_dict):
    """Called when the module is loaded into LLDB."""
    # Register commands
    debugger.HandleCommand(
        'command script add -f lldb_agx_debug.agx_setup agx_setup')
    debugger.HandleCommand(
        'command script add -f lldb_agx_debug.agx_crash_info agx_crash_info')
    debugger.HandleCommand(
        'command script add -f lldb_agx_debug.agx_encoders agx_encoders')
    debugger.HandleCommand(
        'command script add -f lldb_agx_debug.agx_mutex_stats agx_mutex_stats')
    print("AGX Debug commands loaded: agx_setup, agx_crash_info, agx_encoders, agx_mutex_stats")


def agx_setup(debugger, command, result, internal_dict):
    """Set up breakpoints for AGX race condition debugging."""
    target = debugger.GetSelectedTarget()
    if not target:
        result.AppendMessage("Error: No target selected")
        return

    print("=== AGX Driver Debug Setup ===\n")

    # Breakpoint on crash site 1: setComputePipelineState:
    bp1 = target.BreakpointCreateByName("-[AGXG16XFamilyComputeContext setComputePipelineState:]")
    if bp1.IsValid():
        bp1.SetScriptCallbackFunction("lldb_agx_debug.on_set_pipeline")
        print(f"[BP1] setComputePipelineState: (crash site 1) - ID {bp1.GetID()}")
    else:
        # Try symbol regex
        bp1 = target.BreakpointCreateByRegex("setComputePipelineState")
        print(f"[BP1] setComputePipelineState (regex) - ID {bp1.GetID()}")

    # Breakpoint on crash site 2: prepareForEnqueue
    bp2 = target.BreakpointCreateByRegex("prepareForEnqueue")
    if bp2.IsValid():
        bp2.SetScriptCallbackFunction("lldb_agx_debug.on_prepare_enqueue")
        print(f"[BP2] prepareForEnqueue (crash site 2) - ID {bp2.GetID()}")

    # Breakpoint on crash site 3: allocateUSCSpillBuffer
    bp3 = target.BreakpointCreateByRegex("allocateUSCSpillBuffer")
    if bp3.IsValid():
        bp3.SetScriptCallbackFunction("lldb_agx_debug.on_alloc_spill")
        print(f"[BP3] allocateUSCSpillBuffer (crash site 3) - ID {bp3.GetID()}")

    # Breakpoint on encoder creation
    bp4 = target.BreakpointCreateByName("-[AGXG16XFamilyComputeContext init]")
    if bp4.IsValid():
        bp4.SetScriptCallbackFunction("lldb_agx_debug.on_context_init")
        print(f"[BP4] ComputeContext init - ID {bp4.GetID()}")

    # Breakpoint on encoder destruction
    bp5 = target.BreakpointCreateByName("-[AGXG16XFamilyComputeContext dealloc]")
    if bp5.IsValid():
        bp5.SetScriptCallbackFunction("lldb_agx_debug.on_context_dealloc")
        print(f"[BP5] ComputeContext dealloc - ID {bp5.GetID()}")

    # Breakpoint on deferredEndEncoding (teardown)
    bp6 = target.BreakpointCreateByName("-[AGXG16XFamilyComputeContext deferredEndEncoding]")
    if bp6.IsValid():
        bp6.SetScriptCallbackFunction("lldb_agx_debug.on_deferred_end")
        print(f"[BP6] deferredEndEncoding - ID {bp6.GetID()}")

    # Set stop on SIGSEGV/SIGBUS
    debugger.HandleCommand("process handle SIGSEGV -s true -p true -n false")
    debugger.HandleCommand("process handle SIGBUS -s true -p true -n false")
    print("\n[SIGNAL] SIGSEGV/SIGBUS will stop execution")

    print("\n=== Setup Complete ===")
    print("Run your test with: (lldb) run")
    print("On crash, use: (lldb) agx_crash_info")
    result.AppendMessage("AGX debug setup complete")


def on_set_pipeline(frame, bp_loc, dict):
    """Callback when setComputePipelineState: is called."""
    thread = frame.GetThread()
    tid = thread.GetThreadID()

    # Get self (arg0 = x0 on ARM64)
    encoder_addr = frame.FindRegister("x0").GetValueAsUnsigned()

    # Track this encoder
    if encoder_addr not in encoder_contexts:
        encoder_contexts[encoder_addr] = {
            'created_tid': tid,
            'calls': 0
        }
    encoder_contexts[encoder_addr]['calls'] += 1
    encoder_contexts[encoder_addr]['last_tid'] = tid

    # Check for cross-thread access (potential race)
    if encoder_contexts[encoder_addr]['created_tid'] != tid:
        print(f"[WARN] Cross-thread encoder access! encoder={hex(encoder_addr)} "
              f"created_by={encoder_contexts[encoder_addr]['created_tid']} accessed_by={tid}")

    return False  # Continue execution


def on_prepare_enqueue(frame, bp_loc, dict):
    """Callback for prepareForEnqueue (crash site 2)."""
    thread = frame.GetThread()
    tid = thread.GetThreadID()
    print(f"[BP2] prepareForEnqueue tid={tid} frame={frame.GetFunctionName()}")
    return False


def on_alloc_spill(frame, bp_loc, dict):
    """Callback for allocateUSCSpillBuffer (crash site 3)."""
    thread = frame.GetThread()
    tid = thread.GetThreadID()
    print(f"[BP3] allocateUSCSpillBuffer tid={tid}")
    return False


def on_context_init(frame, bp_loc, dict):
    """Callback when a new ComputeContext is created."""
    thread = frame.GetThread()
    tid = thread.GetThreadID()
    encoder_addr = frame.FindRegister("x0").GetValueAsUnsigned()

    encoder_contexts[encoder_addr] = {
        'created_tid': tid,
        'calls': 0,
        'state': 'init'
    }
    print(f"[INIT] ComputeContext {hex(encoder_addr)} created on tid={tid}")
    return False


def on_context_dealloc(frame, bp_loc, dict):
    """Callback when a ComputeContext is deallocated."""
    thread = frame.GetThread()
    tid = thread.GetThreadID()
    encoder_addr = frame.FindRegister("x0").GetValueAsUnsigned()

    if encoder_addr in encoder_contexts:
        ctx = encoder_contexts[encoder_addr]
        if ctx.get('state') == 'in_use':
            print(f"[RACE!] ComputeContext {hex(encoder_addr)} deallocated while in use!")
            print(f"        Created by tid={ctx['created_tid']}, dealloc tid={tid}")
        encoder_contexts[encoder_addr]['state'] = 'dealloc'

    print(f"[DEALLOC] ComputeContext {hex(encoder_addr)} on tid={tid}")
    return False


def on_deferred_end(frame, bp_loc, dict):
    """Callback for deferredEndEncoding (destroys impl)."""
    thread = frame.GetThread()
    tid = thread.GetThreadID()
    encoder_addr = frame.FindRegister("x0").GetValueAsUnsigned()
    print(f"[END] deferredEndEncoding {hex(encoder_addr)} on tid={tid}")

    if encoder_addr in encoder_contexts:
        encoder_contexts[encoder_addr]['state'] = 'ended'
    return False


def agx_crash_info(debugger, command, result, internal_dict):
    """Print detailed info at crash site."""
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetSelectedFrame()

    print("=== AGX Crash Analysis ===\n")

    # Basic crash info
    print(f"Thread: {thread.GetThreadID()}")
    print(f"Stop reason: {thread.GetStopDescription(100)}")
    print(f"Function: {frame.GetFunctionName()}")
    print(f"PC: {hex(frame.GetPC())}")

    # Registers (ARM64)
    print("\n--- Registers ---")
    for i in range(29):
        reg = frame.FindRegister(f"x{i}")
        if reg.IsValid():
            val = reg.GetValueAsUnsigned()
            # Only print non-zero registers
            if val != 0:
                print(f"x{i}: {hex(val)}")

    # Check if crash was NULL dereference
    print("\n--- Crash Address Analysis ---")
    # Try to get fault address from signal info
    stop_desc = thread.GetStopDescription(200)
    if "address=" in stop_desc or "at 0x" in stop_desc:
        print(f"Likely NULL pointer dereference in AGX driver")
        print(f"Known crash offsets: 0x98, 0x184, 0x5c8")

    # Show tracked encoders
    print("\n--- Tracked Encoders ---")
    for addr, ctx in encoder_contexts.items():
        state = ctx.get('state', 'unknown')
        print(f"  {hex(addr)}: state={state} created_tid={ctx.get('created_tid')} "
              f"calls={ctx.get('calls', 0)}")

    # Backtrace
    print("\n--- Backtrace ---")
    for i, frame in enumerate(thread):
        print(f"  #{i}: {frame.GetFunctionName()} + {frame.GetPC() - frame.GetFunction().GetStartAddress().GetLoadAddress(target)}")
        if i > 15:
            print("  ...")
            break

    result.AppendMessage("\nUse 'bt' for full backtrace, 'register read' for all registers")


def agx_encoders(debugger, command, result, internal_dict):
    """List all tracked encoder contexts."""
    print("=== Tracked Encoder Contexts ===\n")

    if not encoder_contexts:
        print("No encoders tracked yet. Run 'agx_setup' and start your test.")
        return

    for addr, ctx in sorted(encoder_contexts.items()):
        state = ctx.get('state', 'unknown')
        created = ctx.get('created_tid', '?')
        last = ctx.get('last_tid', '?')
        calls = ctx.get('calls', 0)
        print(f"{hex(addr)}: state={state:10} created_tid={created} last_tid={last} calls={calls}")


def agx_mutex_stats(debugger, command, result, internal_dict):
    """Show AGX fix mutex statistics if libagx_fix.dylib is loaded."""
    target = debugger.GetSelectedTarget()

    # Look for agx_fix symbols
    acq_sym = target.FindSymbols("g_agx_mutex_acquisitions")
    cont_sym = target.FindSymbols("g_agx_mutex_contentions")

    if acq_sym.IsValid() and cont_sym.IsValid():
        # Read the values (this is simplified - real implementation would read memory)
        print("AGX Fix mutex statistics:")
        print("  (Use 'expr g_agx_mutex_acquisitions' to read actual value)")
        print("  (Use 'expr g_agx_mutex_contentions' to read actual value)")
    else:
        print("libagx_fix.dylib not loaded or symbols not found")
        print("Load with: DYLD_INSERT_LIBRARIES=libagx_fix.dylib")


# Standalone usage
if __name__ == "__main__":
    print(__doc__)
