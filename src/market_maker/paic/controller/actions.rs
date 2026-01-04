//! Impulse action types for PAIC controller.
//!
//! Defines the actions the controller can take for each order.

/// Impulse action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpulseAction {
    /// HOLD: Keep order as-is, priority is valuable
    Hold,

    /// LEAK: Reduce size to lower exposure while keeping queue position
    /// Parameter: new size (as fraction of original)
    Leak,

    /// SHADOW: Update price to track fair price (low-priority chase)
    Shadow,

    /// RESET: Aggressively modify price (abandon queue position)
    Reset,
}

impl ImpulseAction {
    /// Get display name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hold => "HOLD",
            Self::Leak => "LEAK",
            Self::Shadow => "SHADOW",
            Self::Reset => "RESET",
        }
    }

    /// Check if this action modifies price.
    pub fn modifies_price(&self) -> bool {
        matches!(self, Self::Shadow | Self::Reset)
    }

    /// Check if this action modifies size.
    pub fn modifies_size(&self) -> bool {
        matches!(self, Self::Leak)
    }

    /// Get urgency level (0 = no action, 1 = low, 2 = medium, 3 = high).
    pub fn urgency(&self) -> u8 {
        match self {
            Self::Hold => 0,
            Self::Shadow => 1,
            Self::Leak => 2,
            Self::Reset => 3,
        }
    }
}

/// Order action to be executed.
#[derive(Debug, Clone)]
pub enum OrderAction {
    /// No action needed
    None,

    /// Modify price only
    ModifyPrice {
        oid: u64,
        new_price: f64,
    },

    /// Modify size only (reduce)
    ModifySize {
        oid: u64,
        new_size: f64,
    },

    /// Modify both price and size
    ModifyBoth {
        oid: u64,
        new_price: f64,
        new_size: f64,
    },

    /// Cancel order entirely
    Cancel {
        oid: u64,
    },
}

impl OrderAction {
    /// Get order ID if applicable.
    pub fn oid(&self) -> Option<u64> {
        match self {
            Self::None => None,
            Self::ModifyPrice { oid, .. } => Some(*oid),
            Self::ModifySize { oid, .. } => Some(*oid),
            Self::ModifyBoth { oid, .. } => Some(*oid),
            Self::Cancel { oid } => Some(*oid),
        }
    }

    /// Check if this is a no-op.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Full impulse decision with reasoning.
#[derive(Debug, Clone)]
pub struct ImpulseDecision {
    /// Order ID
    pub oid: u64,
    /// Decided action type
    pub action: ImpulseAction,
    /// Concrete action to execute
    pub order_action: OrderAction,
    /// Priority index at decision time
    pub pi: f64,
    /// Drift from fair price in basis points
    pub drift_bps: f64,
    /// Priority premium in basis points
    pub priority_premium_bps: f64,
    /// Modify threshold in basis points
    pub modify_threshold_bps: f64,
    /// Was toxic flow detected?
    pub is_toxic: bool,
    /// Importance score for rate limiting
    pub importance: f64,
}

impl ImpulseDecision {
    /// Create a HOLD decision.
    pub fn hold(oid: u64, pi: f64, drift_bps: f64, priority_premium_bps: f64, threshold_bps: f64) -> Self {
        Self {
            oid,
            action: ImpulseAction::Hold,
            order_action: OrderAction::None,
            pi,
            drift_bps,
            priority_premium_bps,
            modify_threshold_bps: threshold_bps,
            is_toxic: false,
            importance: 0.0,
        }
    }

    /// Create a LEAK decision.
    pub fn leak(
        oid: u64,
        new_size: f64,
        pi: f64,
        drift_bps: f64,
        priority_premium_bps: f64,
        threshold_bps: f64,
    ) -> Self {
        Self {
            oid,
            action: ImpulseAction::Leak,
            order_action: OrderAction::ModifySize { oid, new_size },
            pi,
            drift_bps,
            priority_premium_bps,
            modify_threshold_bps: threshold_bps,
            is_toxic: true,
            importance: drift_bps / threshold_bps,
        }
    }

    /// Create a SHADOW decision.
    pub fn shadow(
        oid: u64,
        new_price: f64,
        pi: f64,
        drift_bps: f64,
        priority_premium_bps: f64,
        threshold_bps: f64,
    ) -> Self {
        Self {
            oid,
            action: ImpulseAction::Shadow,
            order_action: OrderAction::ModifyPrice { oid, new_price },
            pi,
            drift_bps,
            priority_premium_bps,
            modify_threshold_bps: threshold_bps,
            is_toxic: false,
            importance: (drift_bps / threshold_bps).clamp(0.0, 1.0),
        }
    }

    /// Create a RESET decision.
    pub fn reset(
        oid: u64,
        new_price: f64,
        pi: f64,
        drift_bps: f64,
        priority_premium_bps: f64,
        threshold_bps: f64,
    ) -> Self {
        Self {
            oid,
            action: ImpulseAction::Reset,
            order_action: OrderAction::ModifyPrice { oid, new_price },
            pi,
            drift_bps,
            priority_premium_bps,
            modify_threshold_bps: threshold_bps,
            is_toxic: false,
            importance: (drift_bps / threshold_bps).min(2.0),
        }
    }
}
