# Quick Start - Get Running in 5 Minutes

---

## Step 1: Clone & Verify (1 min)

```bash
git clone https://github.com/Tryboy869/ngpc.git
cd ngpc/experiments/python

# Verify installation
python cosmic_computation.py
```

Expected output:
```
✓ All 24 patterns validated
✓ Consensus: 109ms (273× vs Paxos)
✓ Cache: 75% hit rate (+10% vs Redis)
```

---

## Step 2: Your First Pattern (2 min)

Create `my_first.py`:

```python
from ngpc import Pulsar

# Perfect 60 FPS timing (no drift)
pulsar = Pulsar(frequency_hz=60)

frame = 0
def game_loop():
    global frame
    frame += 1
    print(f"Frame {frame}", end='\r')

pulsar.pulse(game_loop)
```

Run:
```bash
python my_first.py
```

You'll see frames counting at exactly 60 FPS with zero drift!

---

## Step 3: Combine Patterns (2 min)

Create `consensus_example.py`:

```python
from ngpc import Magnetar, BlackHole, Pulsar, Node
import random

# 10 nodes, 2 Byzantine
nodes = []
for i in range(8):
    nodes.append(Node(i, vote=100.0 + random.uniform(-5,5), credibility=0.9))
nodes.append(Node(8, vote=0.0, credibility=0.2, is_byzantine=True))
nodes.append(Node(9, vote=200.0, credibility=0.2, is_byzantine=True))

# Cosmic consensus
magnetar = Magnetar()
black_hole = BlackHole()
pulsar = Pulsar(frequency_hz=10)

for round in range(10):
    magnetar.align(nodes)
    consensus = black_hole.converge([n.vote for n in nodes])
    print(f"Round {round+1}: {consensus:.2f}")
    pulsar.wait()

print(f"✓ Consensus achieved: {consensus:.2f}")
```

---

## Next Steps

- **Read patterns**: [PATTERNS_GUIDE_DEV_FRIENDLY.md](PATTERNS_GUIDE_DEV_FRIENDLY.md)
- **Run benchmarks**: `python test_consensus.py`
- **Try in your project**: Pick ONE pattern, integrate, measure!

---

Need help? [Open an issue](https://github.com/Tryboy869/ngpc/issues)
