package handshake

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"
)

// VesselType represents the role of a vessel in the fleet
type VesselType int

const (
	Lighthouse VesselType = iota
	Vessel
	Scout
	Barnacle
	Ghost
)

func (t VesselType) String() string {
	return []string{"Lighthouse", "Vessel", "Scout", "Barnacle", "Ghost"}[t]
}

// VesselStatus represents operational status
type VesselStatus int

const (
	Active VesselStatus = iota
	Dormant
	Maintenance
	Decommissioned
)

func (s VesselStatus) String() string {
	return []string{"Active", "Dormant", "Maintenance", "Decommissioned"}[s]
}

// Beacon is the broadcast identity of a vessel
type Beacon struct {
	Name         string   `json:"name"`
	Type         VesselType `json:"type"`
	Status       VesselStatus `json:"status"`
	RepoURL      string   `json:"repo_url"`
	Capabilities []string `json:"capabilities"`
	Hardware     HardwareSpec `json:"hardware"`
	APIs         []string `json:"apis"`
	FleetRank    int      `json:"fleet_rank"`
	LastSeen     string   `json:"last_seen"`
}

// HardwareSpec describes the vessel's hardware
type HardwareSpec struct {
	CPU      string `json:"cpu"`
	RAM      string `json:"ram"`
	GPU      string `json:"gpu,omitempty"`
	Provider string `json:"provider"`
}

// HandshakeRequest initiates fleet discovery
type HandshakeRequest struct {
	From      Beacon  `json:"from"`
	To        string  `json:"to"`
	Timestamp string  `json:"timestamp"`
	Nonce     string  `json:"nonce"`
}

// HandshakeResponse acknowledges discovery
type HandshakeResponse struct {
	From            Beacon  `json:"from"`
	To              string  `json:"to"`
	SharedCaps      []string `json:"shared_capabilities"`
	NewCaps         []string `json:"new_capabilities"`
	Timestamp       string  `json:"timestamp"`
	RequestNonce    string  `json:"request_nonce"`
}

// NewBeacon creates a new vessel beacon
func NewBeacon(name string, vtype VesselType, repo string) *Beacon {
	return &Beacon{
		Name:     name,
		Type:     vtype,
		Status:   Active,
		RepoURL:  repo,
		LastSeen: time.Now().UTC().Format(time.RFC3339),
	}
}

// Touch updates the last-seen timestamp
func (b *Beacon) Touch() {
	b.LastSeen = time.Now().UTC().Format(time.RFC3339)
}

// HasCapability checks if the vessel has a specific capability
func (b *Beacon) HasCapability(cap string) bool {
	for _, c := range b.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// Handshake performs capability exchange between two vessels
func Handshake(local, remote *Beacon) *HandshakeResponse {
	localSet := make(map[string]bool)
	for _, c := range local.Capabilities {
		localSet[c] = true
	}

	var shared []string
	var newCaps []string
	for _, c := range remote.Capabilities {
		if localSet[c] {
			shared = append(shared, c)
		} else {
			newCaps = append(newCaps, c)
		}
	}

	return &HandshakeResponse{
		From:       *local,
		To:         remote.Name,
		SharedCaps: shared,
		NewCaps:    newCaps,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}
}

// NewHandshakeRequest creates a discovery request
func NewHandshakeRequest(from *Beacon, to string) *HandshakeRequest {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%s:%s:%d", from.Name, to, time.Now().UnixNano())))
	return &HandshakeRequest{
		From:      *from,
		To:        to,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Nonce:     fmt.Sprintf("%x", h.Sum(nil))[:12],
	}
}

// ToJSON serializes a beacon to JSON
func (b *Beacon) ToJSON() (string, error) {
	data, err := json.MarshalIndent(b, "", "  ")
	return string(data), err
}

// BeaconFromJSON deserializes a beacon from JSON
func BeaconFromJSON(j string) (*Beacon, error) {
	var b Beacon
	err := json.Unmarshal([]byte(j), &b)
	return &b, err
}
