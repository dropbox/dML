/-
  MPSVerify.Bridges
  External Tool Bridges for MPS Verification Platform

  This module re-exports all bridges to external verification tools:
  - TLAPlus: TLC model checker output parser
  - TLCRunner: TLC process execution
  - CBMC: CBMC output parser
  - CBMCRunner: CBMC process execution
  - StaticAnalysis: Clang TSA, Facebook Infer
-/

import MPSVerify.Bridges.TLAPlus
import MPSVerify.Bridges.TLCRunner
import MPSVerify.Bridges.CBMC
import MPSVerify.Bridges.CBMCRunner
import MPSVerify.Bridges.StaticAnalysis
