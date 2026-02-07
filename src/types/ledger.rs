//! Ledger and transaction types.

use alloy::primitives::Address;
use serde::{Deserialize, Serialize};

/// User non-funding ledger updates data.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct UserNonFundingLedgerUpdatesData {
    pub is_snapshot: Option<bool>,
    pub user: Address,
    pub non_funding_ledger_updates: Vec<LedgerUpdateData>,
}

/// Ledger update with timestamp and hash.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct LedgerUpdateData {
    pub time: u64,
    pub hash: String,
    pub delta: LedgerUpdate,
}

/// Ledger update types.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "type")]
pub enum LedgerUpdate {
    Deposit(Deposit),
    Withdraw(Withdraw),
    InternalTransfer(InternalTransfer),
    SubAccountTransfer(SubAccountTransfer),
    LedgerLiquidation(LedgerLiquidation),
    VaultDeposit(VaultDelta),
    VaultCreate(VaultDelta),
    VaultDistribution(VaultDelta),
    VaultWithdraw(VaultWithdraw),
    VaultLeaderCommission(VaultLeaderCommission),
    AccountClassTransfer(AccountClassTransfer),
    SpotTransfer(SpotTransferLedger),
    SpotGenesis(SpotGenesis),
    /// HIP token send (e.g., on-chain transfers)
    Send(TokenSend),
    /// Catch-all for unknown ledger update types to prevent deserialization failures
    #[serde(other)]
    Unknown,
}

/// Token send event (HIP transfers, etc.)
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct TokenSend {
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub amount: Option<String>,
    #[serde(default)]
    pub destination: Option<Address>,
    #[serde(default)]
    pub fee: Option<String>,
}

/// Deposit event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Deposit {
    pub usdc: String,
}

/// Withdraw event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Withdraw {
    pub usdc: String,
    pub nonce: u64,
    pub fee: String,
}

/// Internal transfer event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct InternalTransfer {
    pub usdc: String,
    pub user: Address,
    pub destination: Address,
    pub fee: String,
}

/// Sub-account transfer event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct SubAccountTransfer {
    pub usdc: String,
    pub user: Address,
    pub destination: Address,
}

/// Ledger liquidation event.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct LedgerLiquidation {
    pub account_value: u64,
    pub leverage_type: String,
    pub liquidated_positions: Vec<LiquidatedPosition>,
}

/// Liquidated position information.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct LiquidatedPosition {
    pub coin: String,
    pub szi: String,
}

/// Vault delta (deposit/create/distribution).
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct VaultDelta {
    pub vault: Address,
    pub usdc: String,
}

/// Vault withdraw event.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct VaultWithdraw {
    pub vault: Address,
    pub user: Address,
    pub requested_usd: String,
    pub commission: String,
    pub closing_cost: String,
    pub basis: String,
    pub net_withdrawn_usd: String,
}

/// Vault leader commission event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct VaultLeaderCommission {
    pub user: Address,
    pub usdc: String,
}

/// Account class transfer event.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AccountClassTransfer {
    pub usdc: String,
    pub to_perp: bool,
}

/// Spot transfer ledger event.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SpotTransferLedger {
    pub token: String,
    pub amount: String,
    pub usdc_value: String,
    pub user: Address,
    pub destination: Address,
    pub fee: String,
}

/// Spot genesis event.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct SpotGenesis {
    pub token: String,
    pub amount: String,
}
