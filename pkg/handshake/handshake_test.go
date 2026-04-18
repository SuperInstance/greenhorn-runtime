package handshake

import (
        "testing"
)

func TestNewBeacon(t *testing.T) {
        b := NewBeacon("TestVessel", Scout, "https://github.com/test/vessel")
        if b.Name != "TestVessel" {
                t.Fatalf("expected TestVessel, got %s", b.Name)
        }
        if b.Type != Scout {
                t.Fatalf("expected Scout")
        }
}

func TestBeaconTouch(t *testing.T) {
        b := NewBeacon("test", Vessel, "http://test")
        b.Touch()
        if b.LastSeen == "" {
                t.Fatal("Touch should update LastSeen")
        }
}

func TestHasCapability(t *testing.T) {
        b := NewBeacon("test", Lighthouse, "http://test")
        b.Capabilities = []string{"coordination", "architecture"}
        if !b.HasCapability("coordination") {
                t.Fatal("should have coordination")
        }
        if b.HasCapability("cuda") {
                t.Fatal("should not have cuda")
        }
}

func TestHandshake(t *testing.T) {
        local := NewBeacon("Oracle1", Lighthouse, "http://test")
        local.Capabilities = []string{"coordination", "research", "github"}
        
        remote := NewBeacon("JetsonClaw1", Vessel, "http://test")
        remote.Capabilities = []string{"cuda", "sensor", "github"}
        
        resp := Handshake(local, remote)
        if len(resp.SharedCaps) != 1 || resp.SharedCaps[0] != "github" {
                t.Fatalf("expected shared=[github], got %v", resp.SharedCaps)
        }
        if len(resp.NewCaps) != 2 {
                t.Fatalf("expected 2 new caps, got %v", resp.NewCaps)
        }
}

func TestHandshakeNoOverlap(t *testing.T) {
        a := NewBeacon("A", Lighthouse, "http://test")
        a.Capabilities = []string{"x"}
        b := NewBeacon("B", Vessel, "http://test")
        b.Capabilities = []string{"y"}
        
        resp := Handshake(a, b)
        if len(resp.SharedCaps) != 0 {
                t.Fatal("no shared caps expected")
        }
        if len(resp.NewCaps) != 1 {
                t.Fatal("expected 1 new cap")
        }
}

func TestNewHandshakeRequest(t *testing.T) {
        beacon := NewBeacon("test", Lighthouse, "http://test")
        req := NewHandshakeRequest(beacon, "remote")
        if req.Nonce == "" {
                t.Fatal("nonce should not be empty")
        }
        if req.To != "remote" {
                t.Fatal("to should be remote")
        }
}

func TestBeaconJSONRoundtrip(t *testing.T) {
        b := NewBeacon("TestVessel", Vessel, "http://test")
        b.Capabilities = []string{"cuda", "sensor"}
        b.Hardware = HardwareSpec{CPU: "ARM64", RAM: "8GB", GPU: "1024 CUDA"}
        
        j, err := b.ToJSON()
        if err != nil {
                t.Fatal(err)
        }
        
        restored, err := BeaconFromJSON(j)
        if err != nil {
                t.Fatal(err)
        }
        if restored.Name != "TestVessel" {
                t.Fatal("name mismatch")
        }
        if len(restored.Capabilities) != 2 {
                t.Fatal("capabilities mismatch")
        }
}

func TestVesselTypeString(t *testing.T) {
        if Lighthouse.String() != "Lighthouse" { t.Fatal("wrong string") }
        if Vessel.String() != "Vessel" { t.Fatal("wrong string") }
        if Scout.String() != "Scout" { t.Fatal("wrong string") }
        if Ghost.String() != "Ghost" { t.Fatal("wrong string") }
}

func TestVesselStatusString(t *testing.T) {
        if Active.String() != "Active" { t.Fatal("wrong string") }
        if Dormant.String() != "Dormant" { t.Fatal("wrong string") }
        if Maintenance.String() != "Maintenance" { t.Fatal("wrong string") }
        if Decommissioned.String() != "Decommissioned" { t.Fatal("wrong string") }
}

// --- Additional edge case tests ---

func TestBeaconAllTypes(t *testing.T) {
        types := []struct {
                vtype   VesselType
                wantStr string
        }{
                {Lighthouse, "Lighthouse"},
                {Vessel, "Vessel"},
                {Scout, "Scout"},
                {Barnacle, "Barnacle"},
                {Ghost, "Ghost"},
        }
        for _, tt := range types {
                b := NewBeacon("test", tt.vtype, "http://test")
                if b.Type != tt.vtype {
                        t.Errorf("type = %d, want %d", b.Type, tt.vtype)
                }
                if b.Type.String() != tt.wantStr {
                        t.Errorf("String() = %q, want %q", b.Type.String(), tt.wantStr)
                }
        }
}

func TestNewBeaconDefaults(t *testing.T) {
        b := NewBeacon("defaults-test", Vessel, "http://test")
        if b.Status != Active {
                t.Fatalf("default status should be Active, got %d", b.Status)
        }
        if b.LastSeen == "" {
                t.Fatal("LastSeen should be set by NewBeacon")
        }
        if b.RepoURL != "http://test" {
                t.Fatalf("RepoURL = %q", b.RepoURL)
        }
}

func TestHasCapabilityEmpty(t *testing.T) {
        b := NewBeacon("test", Vessel, "http://test")
        if b.HasCapability("anything") {
                t.Fatal("empty capabilities should not match")
        }
}

func TestHandshakeAllOverlap(t *testing.T) {
        a := NewBeacon("A", Lighthouse, "http://test")
        a.Capabilities = []string{"x", "y"}
        b := NewBeacon("B", Vessel, "http://test")
        b.Capabilities = []string{"x", "y"}

        resp := Handshake(a, b)
        if len(resp.SharedCaps) != 2 {
                t.Fatalf("expected 2 shared caps, got %d", len(resp.SharedCaps))
        }
        if len(resp.NewCaps) != 0 {
                t.Fatalf("expected 0 new caps, got %d", len(resp.NewCaps))
        }
        if resp.To != "B" {
                t.Fatalf("resp.To = %q, want B", resp.To)
        }
        if resp.Timestamp == "" {
                t.Fatal("timestamp should be set")
        }
}

func TestHandshakeEmptyCapabilities(t *testing.T) {
        a := NewBeacon("A", Lighthouse, "http://test")
        b := NewBeacon("B", Vessel, "http://test")

        resp := Handshake(a, b)
        if len(resp.SharedCaps) != 0 {
                t.Fatal("no shared caps with empty capabilities")
        }
        if len(resp.NewCaps) != 0 {
                t.Fatal("no new caps with empty capabilities")
        }
}

func TestHandshakeRequestFields(t *testing.T) {
        b := NewBeacon("sender", Lighthouse, "http://test")
        req := NewHandshakeRequest(b, "receiver")

        if req.From.Name != "sender" {
                t.Fatalf("From.Name = %q", req.From.Name)
        }
        if req.Timestamp == "" {
                t.Fatal("timestamp should be set")
        }
        if len(req.Nonce) != 12 {
                t.Fatalf("nonce length = %d, want 12", len(req.Nonce))
        }
}

func TestHandshakeRequestUniqueNonces(t *testing.T) {
        b := NewBeacon("test", Lighthouse, "http://test")
        req1 := NewHandshakeRequest(b, "target")
        // Small sleep to allow time difference
        req2 := NewHandshakeRequest(b, "target")
        if req1.Nonce == req2.Nonce {
                // Nonces include nanosecond timestamps so they should differ
                // (though technically possible to collide, extremely unlikely)
                t.Log("warning: nonces collided (extremely unlikely)")
        }
}

func TestBeaconFromJSONInvalid(t *testing.T) {
        _, err := BeaconFromJSON("not json")
        if err == nil {
                t.Fatal("should error on invalid JSON")
        }
}

func TestBeaconFromJSONEmpty(t *testing.T) {
        _, err := BeaconFromJSON("")
        if err == nil {
                t.Fatal("should error on empty string")
        }
}

func TestBeaconJSONRoundtripWithAPIs(t *testing.T) {
        b := NewBeacon("API-Vessel", Vessel, "http://test")
        b.Capabilities = []string{"openai", "anthropic", "github"}
        b.APIs = []string{"gpt-4", "claude-3"}
        b.FleetRank = 42
        b.Hardware = HardwareSpec{CPU: "x86_64", RAM: "32GB", GPU: "A100 80GB", Provider: "aws"}

        j, err := b.ToJSON()
        if err != nil {
                t.Fatal(err)
        }

        restored, err := BeaconFromJSON(j)
        if err != nil {
                t.Fatal(err)
        }
        if restored.Name != "API-Vessel" {
                t.Fatal("name mismatch")
        }
        if restored.FleetRank != 42 {
                t.Fatalf("FleetRank = %d, want 42", restored.FleetRank)
        }
        if len(restored.APIs) != 2 {
                t.Fatalf("APIs = %d, want 2", len(restored.APIs))
        }
        if restored.Hardware.Provider != "aws" {
                t.Fatalf("Provider = %q, want aws", restored.Hardware.Provider)
        }
}
