package connector

import (
        "reflect"
        "regexp"
        "testing"
)

func TestParseFenceBoardMarkdown(t *testing.T) {
        markdown := `# The Fence Board — Active Work

## Active Fences

### 🎨 fence-0x42: Map 16 Viewpoint Opcodes to Unified ISA
- **Owner:** [oracle1-vessel](https://github.com/SuperInstance/oracle1-vessel)
- **Status:** 🟢 OPEN
- **Hook:** "Nobody has defined these yet. They're reserved for you."
- **Difficulty:** Babel 3/10, Oracle1 7/10, JC1 8/10
- **Reward:** Your name on 16 opcodes every runtime executes

### 🧪 fence-0x44: Benchmark Vocabulary Abstraction Cost
- **Owner:** [oracle1-vessel](https://github.com/SuperInstance/oracle1-vessel)
- **Status:** 🟢 OPEN
- **Hook:** "How much does the vocabulary abstraction actually cost?"
- **Difficulty:** JC1 3/10, Oracle1 5/10
- **Reward:** Data-driven pruning decisions. Numbers, not opinions.

## Claimed

### 🟡 fence-0x43: A2A Signal → FLUX Compiler — Oracle1 🔮 (SHIPPED ✅)
- **Owner:** [oracle1-vessel](https://github.com/SuperInstance/oracle1-vessel)
- **Status:** SHIPPED
- **Hook:** "Babel's Signal JSON compiles to FLUX bytecodes."
- **Reward:** Working compiler pipeline
`

        fences, err := parseFenceBoardMarkdown(markdown)
        if err != nil {
                t.Fatalf("parseFenceBoardMarkdown returned error: %v", err)
        }

        if len(fences) != 3 {
                t.Fatalf("expected 3 fences, got %d", len(fences))
        }

        // Verify fence-0x42
        f42 := fences[0]
        if f42.ID != "fence-0x42" {
                t.Errorf("f42.ID = %q, want %q", f42.ID, "fence-0x42")
        }
        if f42.Title != "Map 16 Viewpoint Opcodes to Unified ISA" {
                t.Errorf("f42.Title = %q", f42.Title)
        }
        if f42.Status != "OPEN" {
                t.Errorf("f42.Status = %q, want %q", f42.Status, "OPEN")
        }
        if f42.Hook != "Nobody has defined these yet. They're reserved for you." {
                t.Errorf("f42.Hook = %q", f42.Hook)
        }
        expectedDiff := map[string]int{"Babel": 3, "Oracle1": 7, "JC1": 8}
        if !reflect.DeepEqual(f42.Difficulty, expectedDiff) {
                t.Errorf("f42.Difficulty = %v, want %v", f42.Difficulty, expectedDiff)
        }

        // Verify fence-0x44
        f44 := fences[1]
        if f44.ID != "fence-0x44" {
                t.Errorf("f44.ID = %q, want %q", f44.ID, "fence-0x44")
        }

        // Verify fence-0x43 (claimed/shipped)
        f43 := fences[2]
        if f43.ID != "fence-0x43" {
                t.Errorf("f43.ID = %q, want %q", f43.ID, "fence-0x43")
        }
        if f43.Status != "SHIPPED" {
                t.Errorf("f43.Status = %q, want %q", f43.Status, "SHIPPED")
        }
}

func TestParseStatus(t *testing.T) {
        tests := []struct {
                input    string
                expected string
        }{
                {"OPEN", "OPEN"},
                {"CLAIMED", "CLAIMED"},
                {"SHIPPED", "SHIPPED"},
                {"IN PROGRESS", "IN PROGRESS"},
                {"unknown", "unknown"},
        }

        for _, tt := range tests {
                result := parseStatus(tt.input)
                if result != tt.expected {
                        t.Errorf("parseStatus(%q) = %q, want %q", tt.input, result, tt.expected)
                }
        }
}

func TestParseDifficulty(t *testing.T) {
        re := regexp.MustCompile(`(\w+)\s+(\d+)/10`)
        result := parseDifficulty("Babel 3/10, Oracle1 7/10", re)

        if len(result) != 2 {
                t.Fatalf("expected 2 entries, got %d", len(result))
        }
        if result["Babel"] != 3 {
                t.Errorf("Babel = %d, want 3", result["Babel"])
        }
        if result["Oracle1"] != 7 {
                t.Errorf("Oracle1 = %d, want 7", result["Oracle1"])
        }
}

func TestParseEmptyMarkdown(t *testing.T) {
        fences, err := parseFenceBoardMarkdown("# Just a header\n\nNo fences here.\n")
        if err != nil {
                t.Fatalf("unexpected error: %v", err)
        }
        if len(fences) != 0 {
                t.Errorf("expected 0 fences, got %d", len(fences))
        }
}
