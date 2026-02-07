//! Safety auditing and state reconciliation.
//!
//! This module provides periodic safety checks and state reconciliation
//! between local tracking and exchange state.

mod auditor;

pub use auditor::{AuditResult, SafetyAuditor};
