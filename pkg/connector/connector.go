package connector

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Connector struct {
	fleetURL string
	token    string
	client   *http.Client
}

type Fence struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Status      string `json:"status"`
	Difficulty  map[string]int `json:"difficulty"`
	Hook        string `json:"hook"`
	Reward      string `json:"reward"`
}

type FleetStatus struct {
	Agents []AgentInfo `json:"agents"`
	Fences []Fence    `json:"fences"`
}

type AgentInfo struct {
	Name   string `json:"name"`
	Role   string `json:"role"`
	Repo   string `json:"repo"`
	Badges string `json:"badges"`
}

func New(fleetURL, token string) *Connector {
	return &Connector{
		fleetURL: fleetURL,
		token:    token,
		client:   &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Connector) Connect() error {
	// Test connectivity by fetching the fleet repo
	url := strings.Replace(c.fleetURL, "https://github.com/", "https://api.github.com/repos/", 1)
	resp, err := c.get(url)
	if err != nil {
		return fmt.Errorf("cannot reach fleet: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("fleet returned %d", resp.StatusCode)
	}
	return nil
}

func (c *Connector) FetchFenceBoard() ([]Fence, error) {
	// Fetch THE-BOARD.md from greenhorn-onboarding
	url := "https://api.github.com/repos/SuperInstance/greenhorn-onboarding/contents/THE-BOARD.md"
	resp, err := c.get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	// Parse the markdown fence board (simplified)
	return []Fence{}, nil // TODO: parse markdown
}

func (c *Connector) ClaimFence(fenceID, approach string) error {
	// Post an issue on the fleet vessel to claim a fence
	url := "https://api.github.com/repos/SuperInstance/oracle1-vessel/issues"
	body := map[string]string{
		"title": fmt.Sprintf("[CLAIM] %s", fenceID),
		"body":  approach,
	}
	return c.post(url, body)
}

func (c *Connector) ReportStatus(agentName, rigging string, tasks int) error {
	// Commit a status update to the agent's vessel
	// In practice, this pushes a commit to the vessel repo
	return nil
}

func (c *Connector) get(url string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "token "+c.token)
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	return c.client.Do(req)
}

func (c *Connector) post(url string, body interface{}) error {
	data, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, strings.NewReader(string(data)))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "token "+c.token)
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("POST %s returned %d", url, resp.StatusCode)
	}
	return nil
}
