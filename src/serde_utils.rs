//! Shared serialization utilities for the Hyperliquid SDK.

use alloy::{dyn_abi::Eip712Domain, primitives::Address, sol_types::eip712_domain};
use serde::{ser::SerializeStruct, Serializer};

/// Create the EIP-712 domain for Hyperliquid transactions.
pub fn eip712_domain_for_chain(chain_id: u64) -> Eip712Domain {
    eip712_domain! {
        name: "HyperliquidSignTransaction",
        version: "1",
        chain_id: chain_id,
        verifying_contract: Address::ZERO,
    }
}

/// Serialize a u64 as a hex string prefixed with "0x".
/// Used for signature_chain_id fields in EIP-712 structs.
pub fn serialize_hex<S>(val: &u64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(&format!("0x{val:x}"))
}

/// Serialize an Alloy Signature into the {r, s, v} format expected by Hyperliquid API.
pub fn serialize_signature<S>(
    sig: &alloy::primitives::Signature,
    s: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut state = s.serialize_struct("Signature", 3)?;
    state.serialize_field("r", &sig.r())?;
    state.serialize_field("s", &sig.s())?;
    state.serialize_field("v", &(27 + sig.v() as u64))?;
    state.end()
}

/// Get the Hyperliquid chain name string based on network.
pub fn hyperliquid_chain(is_mainnet: bool) -> &'static str {
    if is_mainnet {
        "Mainnet"
    } else {
        "Testnet"
    }
}
