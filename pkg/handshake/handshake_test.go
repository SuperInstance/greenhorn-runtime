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
	if Decommissioned.String() != "Decommissioned" { t.Fatal("wrong string") }
}
