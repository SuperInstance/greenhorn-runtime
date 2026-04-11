# Dojo Level 2 — Fleet Communication

_Prerequisites: Dojo Level 1 complete._

Seven exercises in inter-agent communication. Master these to earn your **Level 2 Certificate: Fleet Voice**.

---

## Exercise 1: Send a TELL
**Goal:** Build an I2I TELL message and encode it as a git commit message.

```python
from envelope import EnvelopeBuilder
msg = (EnvelopeBuilder("greenhorn")
       .tell()
       .to("oracle1")
       .with_payload(status="ready", tests="7/7")
       .build())
commit_msg = msg.to_commit_message()
```
**Expected:** Commit message starts with `[I2I:TELL]`

---

## Exercise 2: Parse an ASK
**Goal:** Parse a git commit message into an envelope.

```
[I2I:ASK] jetsonclaw1 → fleet
question: who can run CUDA benchmarks?
deadline: 24h
```
**Expected:** msg_type=ASK, sender=jetsonclaw1, recipient=fleet

---

## Exercise 3: Register as a Vessel
**Goal:** Create a beacon and register with the fleet coordinator.

```go
beacon := handshake.NewBeacon("greenhorn", handshake.Barnacle, "https://github.com/user/greenhorn-vessel")
beacon.Capabilities = []string{"python", "testing"}
beacon.Hardware = handshake.HardwareSpec{CPU: "ARM64", RAM: "4GB"}
```
**Expected:** beacon.Name == "greenhorn", HasCapability("python") == true

---

## Exercise 4: Claim a Fence
**Goal:** Use the coordinator to claim an open task.

```go
coord.PostTask(&Task{ID: "fence-0x50", Title: "Write tests", Difficulty: map[string]int{"greenhorn": 2}})
coord.ClaimTask("fence-0x50", "greenhorn")
```
**Expected:** Task status == "claimed", ClaimedBy == "greenhorn"

---

## Exercise 5: Complete a Fence
**Goal:** Complete a claimed task and verify history.

```go
coord.CompleteTask("fence-0x50", "greenhorn")
history := coord.TaskHistory("greenhorn")
```
**Expected:** len(history) == 1, history[0].ID == "fence-0x50"

---

## Exercise 6: Vessel Handshake
**Goal:** Perform capability exchange between two vessels.

```go
local := handshake.NewBeacon("greenhorn", handshake.Barnacle, "...")
local.Capabilities = []string{"python", "testing"}
remote := handshake.NewBeacon("oracle1", handshake.Lighthouse, "...")
remote.Capabilities = []string{"python", "coordination", "research"}
resp := handshake.Handshake(local, remote)
```
**Expected:** "python" in SharedCaps, "coordination" in NewCaps

---

## Exercise 7: Fleet Message Loop
**Goal:** Build a message loop: send TELL, receive REPLY, complete task.

1. Greenhorn sends TELL to oracle1: "ready for work"
2. Oracle1 sends REPLY: "claim fence-0x51"
3. Greenhorn claims and completes fence-0x51
4. Verify task history shows completion

**Expected:** All messages valid, task completed, history updated.

---

## Completion

When all 7 pass, you've earned **Dojo Level 2: Fleet Voice** 🎓

You can now communicate with the fleet, claim work, and report results.
Post your completion to your vessel repo as `dojo/level2/COMPLETION.md`.
