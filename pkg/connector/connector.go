package connector

import (
        "bufio"
        "encoding/base64"
        "encoding/json"
        "fmt"
        "net/http"
        "regexp"
        "strconv"
        "strings"
        "time"
)

type Connector struct {
        fleetURL string
        token    string
        client   *http.Client
}

type Fence struct {
        ID          string          `json:"id"`
        Title       string          `json:"title"`
        Status      string          `json:"status"`
        Owner       string          `json:"owner,omitempty"`
        Difficulty  map[string]int  `json:"difficulty,omitempty"`
        Hook        string          `json:"hook,omitempty"`
        Reward      string          `json:"reward,omitempty"`
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
        // Fetch THE-BOARD.md from greenhorn-onboarding via GitHub API
        url := "https://api.github.com/repos/SuperInstance/greenhorn-onboarding/contents/THE-BOARD.md"
        resp, err := c.get(url)
        if err != nil {
                return nil, fmt.Errorf("fetch board: %w", err)
        }
        defer resp.Body.Close()

        if resp.StatusCode != 200 {
                return nil, fmt.Errorf("fetch board: HTTP %d", resp.StatusCode)
        }

        // GitHub Contents API returns {"content": "<base64>", "encoding": "base64"}
        var contents struct {
                Content  string `json:"content"`
                Encoding string `json:"encoding"`
        }
        if err := json.NewDecoder(resp.Body).Decode(&contents); err != nil {
                return nil, fmt.Errorf("decode response: %w", err)
        }

        // Decode base64 content
        var markdown string
        if contents.Encoding == "base64" {
                decoded, err := base64.StdEncoding.DecodeString(
                        strings.ReplaceAll(contents.Content, "\n", ""),
                )
                if err != nil {
                        return nil, fmt.Errorf("decode base64: %w", err)
                }
                markdown = string(decoded)
        } else {
                markdown = contents.Content
        }

        return parseFenceBoardMarkdown(markdown)
}

// parseFenceBoardMarkdown parses THE-BOARD.md markdown into structured Fence objects.
//
// Expected format:
//
//      ### 🎨 fence-0x42: Map 16 Viewpoint Opcodes to Unified ISA
//      - **Owner:** [oracle1-vessel](https://github.com/...)
//      - **Status:** 🟢 OPEN
//      - **Hook:** "Nobody has defined these yet."
//      - **Difficulty:** Babel 3/10, Oracle1 7/10
//      - **Reward:** Your name on 16 opcodes
func parseFenceBoardMarkdown(markdown string) ([]Fence, error) {
        var fences []Fence

        // Regex to match fence headers: ### emoji fence-0xNN: Title
        headerRe := regexp.MustCompile(`^###\s+[^\s]+\s+(fence-0x[0-9A-Fa-f]+):\s+(.+)$`)

        // Regex to match fence fields: - **Key:** Value
        fieldRe := regexp.MustCompile(`^-\s+\*\*(\w[\w\s]*\w|\w+)\*\*:\s*(.+)$`)

        // Regex to extract difficulty: "Babel 3/10, Oracle1 7/10"
        diffRe := regexp.MustCompile(`(\w+)\s+(\d+)/10`)

        scanner := bufio.NewScanner(strings.NewReader(markdown))
        var current *Fence

        for scanner.Scan() {
                line := strings.TrimSpace(scanner.Text())

                // Check for fence header
                if matches := headerRe.FindStringSubmatch(line); matches != nil {
                        // Save previous fence
                        if current != nil {
                                fences = append(fences, *current)
                        }
                        current = &Fence{
                                ID:     matches[1],
                                Title:  matches[2],
                                Status: "unknown",
                        }
                        continue
                }

                if current == nil {
                        continue
                }

                // Check for field lines
                if matches := fieldRe.FindStringSubmatch(line); matches != nil {
                        key := strings.TrimSpace(matches[1])
                        value := strings.TrimSpace(matches[2])

                        switch strings.ToLower(key) {
                        case "owner":
                                current.Owner = value
                        case "status":
                                current.Status = parseStatus(value)
                        case "hook":
                                // Strip surrounding quotes
                                current.Hook = strings.Trim(value, "\"")
                        case "difficulty":
                                current.Difficulty = parseDifficulty(value, diffRe)
                        case "reward":
                                current.Reward = value
                        }
                }
        }

        // Don't forget the last fence
        if current != nil {
                fences = append(fences, *current)
        }

        // Filter out section headers (Claimed, How to Claim, etc.)
        var filtered []Fence
        for _, f := range fences {
                if strings.HasPrefix(f.ID, "fence-0x") || strings.HasPrefix(f.ID, "fence-") {
                        filtered = append(filtered, f)
                }
        }

        return filtered, scanner.Err()
}

// parseStatus extracts the status string from markdown emoji+text.
// Input: "🟢 OPEN" -> "OPEN"
// Input: "🟡 CLAIMED" -> "CLAIMED"
// Input: "✅ SHIPPED" -> "SHIPPED"
func parseStatus(raw string) string {
        // Strip leading emoji and whitespace
        raw = strings.TrimSpace(raw)
        // Remove common emoji: 🟢 🟡 🔴 ✅ ⚡ 🔮 🌐
        status := regexp.MustCompile(`[^\w\s]`).ReplaceAllString(raw, "")
        return strings.TrimSpace(status)
}

// parseDifficulty extracts difficulty ratings from strings like "Babel 3/10, Oracle1 7/10"
func parseDifficulty(raw string, re *regexp.Regexp) map[string]int {
        diff := make(map[string]int)
        matches := re.FindAllStringSubmatch(raw, -1)
        for _, m := range matches {
                name := m[1]
                rating, err := strconv.Atoi(m[2])
                if err == nil {
                        diff[name] = rating
                }
        }
        return diff
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
