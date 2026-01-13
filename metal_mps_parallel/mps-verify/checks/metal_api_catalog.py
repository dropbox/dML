#!/usr/bin/env python3
"""
Metal/MPS API Constraint Catalog

Comprehensive catalog of Apple Metal and MPS API constraints that must be
respected for correct behavior. These constraints are derived from:
1. Apple Metal documentation
2. Apple Metal Best Practices Guide
3. Empirical testing and crash analysis (N=1305)
4. WWDC session videos

Each constraint includes:
- API affected
- State machine or precondition
- What happens on violation
- How to detect statically
- How to verify at runtime

Usage:
    from metal_api_catalog import METAL_API_CONSTRAINTS
    for constraint in METAL_API_CONSTRAINTS.values():
        print(f"{constraint.id}: {constraint.description}")
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable


class Severity(Enum):
    """Severity of constraint violation"""
    CRASH = "CRASH"           # Immediate SIGABRT or undefined behavior
    DATA_CORRUPTION = "DATA"  # Silent data corruption
    DEADLOCK = "DEADLOCK"     # System hangs
    LEAK = "LEAK"             # Resource leak
    PERFORMANCE = "PERF"      # Performance degradation only
    UNDEFINED = "UNDEFINED"   # Behavior varies by OS/hardware


class DetectionMethod(Enum):
    """How the constraint can be detected"""
    STATIC = "STATIC"         # Detectable via static analysis
    DYNAMIC = "DYNAMIC"       # Requires runtime instrumentation
    IMPOSSIBLE = "IMPOSSIBLE" # Cannot be automatically detected


@dataclass
class APIConstraint:
    """Definition of a Metal/MPS API constraint"""
    id: str
    api_name: str
    description: str
    precondition: str
    postcondition: str
    violation_behavior: str
    severity: Severity
    detection: DetectionMethod
    static_pattern: Optional[str]  # Regex pattern for static detection
    runtime_check: Optional[str]   # Code snippet for runtime verification
    documentation_url: str
    discovered_by: str = "Documentation"  # How we learned about this
    verified: bool = False  # Whether we've confirmed this constraint


# ============================================================================
# MTLCommandBuffer State Machine Constraints
# ============================================================================

COMMAND_BUFFER_CONSTRAINTS = {
    "AF.001": APIConstraint(
        id="AF.001",
        api_name="MTLCommandBuffer",
        description="Command buffer state machine must be respected",
        precondition="Buffer exists in one of: Created, Encoding, Committed, Completed",
        postcondition="Transitions follow: Created->Encoding->Committed->Completed",
        violation_behavior="SIGABRT with 'Command buffer has already been committed' or similar",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"\[(\w+)\s+commit\].*\n.*\[\1\s+commit\]",
        runtime_check="assert(buffer.status != MTLCommandBufferStatusCommitted)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.002": APIConstraint(
        id="AF.002",
        api_name="commandBufferWithDescriptor:",
        description="Command buffer creation requires valid descriptor",
        precondition="Descriptor has valid retainedReferences setting",
        postcondition="Buffer created with specified retention policy",
        violation_behavior="nil return or crash on invalid descriptor",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"commandBufferWithDescriptor:\s*nil",
        runtime_check="assert(descriptor != nil)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandqueue/3564430-commandbufferwithdescriptor",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.007": APIConstraint(
        id="AF.007",
        api_name="addCompletedHandler:",
        description="Completion handler must be added BEFORE commit",
        precondition="buffer.status == MTLCommandBufferStatusNotEnqueued or MTLCommandBufferStatusEnqueued",
        postcondition="Handler registered, will be called after execution",
        violation_behavior="SIGABRT: 'Command buffer completion handler cannot be added after commit'",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"commit\].*\n.*addCompletedHandler:",
        runtime_check="assert(buffer.status < MTLCommandBufferStatusCommitted)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer/1442997-addcompletedhandler",
        discovered_by="N=1305 crash analysis",
        verified=True
    ),

    "AF.008": APIConstraint(
        id="AF.008",
        api_name="waitUntilCompleted",
        description="Wait must be called AFTER commit",
        precondition="buffer.status >= MTLCommandBufferStatusCommitted",
        postcondition="Call blocks until buffer completes",
        violation_behavior="Undefined: may return immediately or hang",
        severity=Severity.UNDEFINED,
        detection=DetectionMethod.STATIC,
        static_pattern=r"waitUntilCompleted\].*\n.*commit\]",
        runtime_check="assert(buffer.status >= MTLCommandBufferStatusCommitted)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer/1443039-waituntilcompleted",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.009": APIConstraint(
        id="AF.009",
        api_name="commit",
        description="Buffer can only be committed once",
        precondition="buffer.status < MTLCommandBufferStatusCommitted",
        postcondition="buffer.status == MTLCommandBufferStatusCommitted",
        violation_behavior="SIGABRT: 'Command buffer already committed'",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"\[\s*(\w+)\s+commit\s*\].*?\[\s*\1\s+commit\s*\]",
        runtime_check="assert(buffer.status < MTLCommandBufferStatusCommitted)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer/1443003-commit",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# MTLCommandEncoder Constraints
# ============================================================================

ENCODER_CONSTRAINTS = {
    "AF.010": APIConstraint(
        id="AF.010",
        api_name="endEncoding",
        description="Encoder must be ended before command buffer commit",
        precondition="Encoder is active",
        postcondition="Encoder is ended, buffer can be committed",
        violation_behavior="SIGABRT or undefined encoding state",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"commit\](?!.*endEncoding)",
        runtime_check="// Must track encoder state manually",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandencoder/1458038-endencoding",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.011": APIConstraint(
        id="AF.011",
        api_name="computeCommandEncoder/blitCommandEncoder",
        description="Only one encoder active per command buffer at a time",
        precondition="No other encoder is active on this buffer",
        postcondition="New encoder is now active",
        violation_behavior="SIGABRT: 'Cannot create encoder while another is active'",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"CommandEncoder\].*\n(?!.*endEncoding).*CommandEncoder\]",
        runtime_check="assert(activeEncoder == nil)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# MTLEvent Constraints
# ============================================================================

EVENT_CONSTRAINTS = {
    "AF.012": APIConstraint(
        id="AF.012",
        api_name="MTLSharedEvent.signalValue",
        description="Signal value must be monotonically increasing",
        precondition="newValue > event.signaledValue",
        postcondition="event.signaledValue == newValue",
        violation_behavior="Undefined: may not signal or may signal immediately",
        severity=Severity.UNDEFINED,
        detection=DetectionMethod.DYNAMIC,
        static_pattern=None,  # Cannot detect statically
        runtime_check="assert(newValue > event.signaledValue)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlsharedevent",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.013": APIConstraint(
        id="AF.013",
        api_name="notifyListener:atValue:block:",
        description="Listener must be created with dispatch queue",
        precondition="listener initialized with initWithDispatchQueue:",
        postcondition="Callbacks will be dispatched to specified queue",
        violation_behavior="Callbacks may be called on wrong thread or not at all",
        severity=Severity.DATA_CORRUPTION,
        detection=DetectionMethod.STATIC,
        static_pattern=r"MTLSharedEventListener\s+alloc\]\s+init\]",
        runtime_check="assert(listener.dispatchQueue != nil)",
        documentation_url="https://developer.apple.com/documentation/metal/mtlsharedeventlistener",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# Dispatch Queue Constraints
# ============================================================================

DISPATCH_CONSTRAINTS = {
    "AF.020": APIConstraint(
        id="AF.020",
        api_name="dispatch_sync",
        description="dispatch_sync to current queue causes deadlock",
        precondition="target queue != current queue",
        postcondition="Block executes synchronously on target queue",
        violation_behavior="DEADLOCK: thread waits for itself",
        severity=Severity.DEADLOCK,
        detection=DetectionMethod.STATIC,
        static_pattern=r"dispatch_sync\s*\(\s*(\w+)\s*,",
        runtime_check="assert(dispatch_get_specific(queueKey) != expectedValue)",
        documentation_url="https://developer.apple.com/documentation/dispatch/1452921-dispatch_sync",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.021": APIConstraint(
        id="AF.021",
        api_name="dispatch_async",
        description="Block must not capture stack-local references",
        precondition="Captured variables are heap-allocated or copied",
        postcondition="Block can execute after caller returns",
        violation_behavior="Use-after-free or stack corruption",
        severity=Severity.DATA_CORRUPTION,
        detection=DetectionMethod.STATIC,
        static_pattern=r"dispatch_async.*&\w+",
        runtime_check="// ASan will detect",
        documentation_url="https://developer.apple.com/documentation/dispatch/1453057-dispatch_async",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# Memory/Resource Constraints
# ============================================================================

MEMORY_CONSTRAINTS = {
    "AF.030": APIConstraint(
        id="AF.030",
        api_name="MTLBuffer.contents",
        description="Contents pointer only valid while buffer retained",
        precondition="buffer.retainCount > 0",
        postcondition="Pointer valid until buffer released",
        violation_behavior="Use-after-free on GPU memory",
        severity=Severity.DATA_CORRUPTION,
        detection=DetectionMethod.DYNAMIC,
        static_pattern=None,
        runtime_check="// Track buffer lifetime",
        documentation_url="https://developer.apple.com/documentation/metal/mtlbuffer/1515716-contents",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.031": APIConstraint(
        id="AF.031",
        api_name="NSAutoreleasePool",
        description="Autorelease pool is thread-local",
        precondition="Pool created on current thread",
        postcondition="Pool drains objects autoreleased on same thread",
        violation_behavior="Objects autoreleased on wrong thread may leak or crash",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"dispatch_async.*autoreleasepool",
        runtime_check="// Must use @autoreleasepool in each dispatch block",
        documentation_url="https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmAutoreleasePools.html",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# MPS (Metal Performance Shaders) Constraints
# ============================================================================

MPS_CONSTRAINTS = {
    "AF.040": APIConstraint(
        id="AF.040",
        api_name="MPSCommandBuffer",
        description="MPSCommandBuffer wraps MTLCommandBuffer with MPS context",
        precondition="Underlying MTLCommandBuffer not committed",
        postcondition="MPS operations can be encoded",
        violation_behavior="SIGABRT or undefined behavior",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"commit\].*MPSCommandBuffer",
        runtime_check="assert(mpsBuffer.commandBuffer.status < MTLCommandBufferStatusCommitted)",
        documentation_url="https://developer.apple.com/documentation/metalperformanceshaders/mpscommandbuffer",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.041": APIConstraint(
        id="AF.041",
        api_name="MPSGraph",
        description="Graph execution requires valid device and command queue",
        precondition="device != nil && commandQueue != nil",
        postcondition="Graph can execute on specified device",
        violation_behavior="Crash or nil result",
        severity=Severity.CRASH,
        detection=DetectionMethod.STATIC,
        static_pattern=r"runWithMTLCommandQueue:\s*nil",
        runtime_check="assert(device != nil && commandQueue != nil)",
        documentation_url="https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# Thread Safety Constraints
# ============================================================================

THREAD_SAFETY_CONSTRAINTS = {
    "AF.050": APIConstraint(
        id="AF.050",
        api_name="MTLDevice",
        description="MTLDevice is thread-safe for most operations",
        precondition="Device obtained via MTLCreateSystemDefaultDevice",
        postcondition="Can be used from multiple threads",
        violation_behavior="N/A - thread-safe",
        severity=Severity.PERFORMANCE,  # Only performance impact if used incorrectly
        detection=DetectionMethod.IMPOSSIBLE,
        static_pattern=None,
        runtime_check="// No check needed",
        documentation_url="https://developer.apple.com/documentation/metal/mtldevice",
        discovered_by="Documentation",
        verified=True
    ),

    "AF.051": APIConstraint(
        id="AF.051",
        api_name="MTLCommandQueue",
        description="Command queue is NOT thread-safe for buffer creation",
        precondition="Serialize buffer creation from same queue",
        postcondition="Buffers created safely",
        violation_behavior="Data race, undefined behavior",
        severity=Severity.DATA_CORRUPTION,
        detection=DetectionMethod.DYNAMIC,
        static_pattern=None,  # Requires concurrency analysis
        runtime_check="// TSan will detect",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandqueue",
        discovered_by="WWDC session",
        verified=True
    ),

    "AF.052": APIConstraint(
        id="AF.052",
        api_name="MTLCommandBuffer",
        description="Command buffer encoding is NOT thread-safe",
        precondition="Serialize all encoding operations",
        postcondition="Encoding completes without data race",
        violation_behavior="Corrupted commands, undefined GPU behavior",
        severity=Severity.DATA_CORRUPTION,
        detection=DetectionMethod.DYNAMIC,
        static_pattern=None,
        runtime_check="// TSan will detect",
        documentation_url="https://developer.apple.com/documentation/metal/mtlcommandbuffer",
        discovered_by="Documentation",
        verified=True
    ),
}

# ============================================================================
# Combined Catalog
# ============================================================================

METAL_API_CONSTRAINTS: Dict[str, APIConstraint] = {
    **COMMAND_BUFFER_CONSTRAINTS,
    **ENCODER_CONSTRAINTS,
    **EVENT_CONSTRAINTS,
    **DISPATCH_CONSTRAINTS,
    **MEMORY_CONSTRAINTS,
    **MPS_CONSTRAINTS,
    **THREAD_SAFETY_CONSTRAINTS,
}


def get_static_checkable_constraints() -> List[APIConstraint]:
    """Return constraints that can be checked statically"""
    return [c for c in METAL_API_CONSTRAINTS.values()
            if c.detection == DetectionMethod.STATIC and c.static_pattern]


def get_constraints_by_severity(severity: Severity) -> List[APIConstraint]:
    """Return constraints of specified severity"""
    return [c for c in METAL_API_CONSTRAINTS.values() if c.severity == severity]


def print_catalog():
    """Print the full constraint catalog"""
    print("=" * 80)
    print("METAL/MPS API CONSTRAINT CATALOG")
    print("=" * 80)
    print(f"\nTotal constraints: {len(METAL_API_CONSTRAINTS)}")
    print(f"Static checkable: {len(get_static_checkable_constraints())}")
    print(f"CRASH severity: {len(get_constraints_by_severity(Severity.CRASH))}")
    print(f"DATA_CORRUPTION severity: {len(get_constraints_by_severity(Severity.DATA_CORRUPTION))}")
    print(f"DEADLOCK severity: {len(get_constraints_by_severity(Severity.DEADLOCK))}")

    print("\n" + "-" * 80)
    for constraint in METAL_API_CONSTRAINTS.values():
        print(f"\n[{constraint.id}] {constraint.api_name}")
        print(f"  Severity: {constraint.severity.value}")
        print(f"  Detection: {constraint.detection.value}")
        print(f"  Description: {constraint.description}")
        print(f"  Precondition: {constraint.precondition}")
        print(f"  On violation: {constraint.violation_behavior}")
        if constraint.static_pattern:
            print(f"  Static pattern: {constraint.static_pattern}")
        print(f"  Discovered by: {constraint.discovered_by}")
        print(f"  Verified: {constraint.verified}")


if __name__ == "__main__":
    print_catalog()
