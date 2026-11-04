"""Generate the Kaggle Writeup (Paper Track) from the paper's data-driven macros."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/arc_prize_2026_submission'


def macros():
    vals = {}
    for name in ['numbers.tex', 'numbers_extra.tex']:
        for k, v in re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', (OUT / name).read_text()):
            vals[k] = v
    return vals


TEMPLATE = '''# Out of Time: Where Test-Time-Trained ARC-AGI-2 Solvers Lose Their Answers

**Subtitle:** A one-line decoding bound explains missing candidates; length-aware decoding recovers them at equal cost.

**Author:** Jingtao Huang (Kaggle: huangjt66)

**Paper (PDF):** attached, `main.pdf` · **Code:** https://github.com/huanji2tong/arc-prize-2026-out-of-time (MIT-0)

---

## Abstract

Before a model can be right about an ARC-AGI-2 task, it has to finish writing the answer. Replaying the frozen candidate pools of a working Kaggle entry on {ConfOutputs} held-out outputs, eight different selectors solve the same {ConfSolved}, and only {ConfOracle} pools contain the answer at all: generation, not selection, sets the ceiling. An h×w grid takes hw+h sequential decoding steps, and splitting a fixed budget evenly across view batches can leave no batch enough time for a long answer. None of the {Unreach} outputs that violate this bound has a correct candidate, and under the same budget {EvalUnreachPct}% of all ARC-AGI-2 public evaluation outputs cannot be decoded at all. **Length-aware decoding** (an exact, recursion-free reimplementation of the solver's DFS plus a first-descent schedule) reduces empty candidate pools from {PilotEmptyCtl}/{PilotN} to {PilotEmptyLA}/{PilotN} and solves {PilotSolvedLA} outputs the original search cannot reach, with the same model, the same 180 seconds, and no losses.{CONF_SENTENCE}

![Figure 1](figures/teaser.png)

## 1. Introduction

ARC-AGI measures how efficiently a system acquires a new skill from a few examples. The strongest open approach for ARC-AGI-2 adapts a language model to each task (test-time training), decodes candidate grids with a probability-thresholded depth-first search over augmented views, and rescores them to pick two guesses (Franzen et al., 2025; NVARC, 2025). We asked where such an entry loses its answers. Saved candidate pools answer this exactly: if the correct grid is in the pool, the loss belongs to selection; otherwise, to generation.

## 2. Prior work

Test-time training for ARC (Akyürek et al.; Cole & Osman), DFS sampling with product-of-experts rescoring (Franzen et al.), synthetic-data training (NVARC), and speculative DFS speedups (the ARChitects, 2025). Our contribution is complementary to faster kernels: we show how the division of a fixed budget among views interacts with answer length to decide which outputs can receive any candidate.

## 3. Approach

* **Coverage–selection decomposition.** pass@2 = coverage × selection, measured by replaying eight selectors on frozen pools; occurrence-count rankings are provably sensitive to duplicated records, view-based ones are not.
* **The decoding clock (Proposition 2).** Under lockstep threshold DFS with t seconds per call, a batch with slice τ can record an L-token candidate only if L·t ≤ τ − prefill. With equal slicing τ = B/(4m).
* **Length-aware decoding.** (i) An iterative DFS that reproduces the parent search exactly (identical suffixes, scores and call counts in 200 tests) without recursion-depth limits and with ARC-only host transfers. (ii) A per-output ledger plus *first-descent extension*: a view may run past its slice only while it is still on its first greedy descent and nothing has finished. Outputs whose first view already succeeds are never extended.

## 4. Results

| Entry (Kaggle ARC-AGI-2, public leaderboard) | Status | Score |
|---|---|---|
| Upstream notebook V1 (Rudra) | author | 33.89 |
| Our V1 reproduction | in competition | 30.97 |
| B0: 360 s decoding, reference selector | late | {ScoreBZero} |
| S1: B0 + NLL-only selector | late | {ScoreSOne} |
| S2: B0 + anchored coverage selector | late | {ScoreSTwo} |
| **L1: B0 + length-aware decoding** | late | {ScoreLOne} |

* **Selection is near its ceiling:** all eight selectors solve {ConfSolved}/{ConfOutputs}; only {ConfMissed} correct grids are left unselected. Both selector changes scored below B0 on the hidden set.
* **The clock explains missing candidates:** the {Unreach} outputs beyond one slice's greedy reach have zero correct candidates; the {Reach} within reach hold all {ConfOracle}. {EvalUnreach} of {EvalOutputs} public evaluation outputs are undecodable under a 360 s budget.
* **Null results, re-read:** on our weight-training panel only {TrainReach}/{TrainN} outputs fit a slice, so five training/decoding interventions could change at most two outputs.
* **Matched pilot (same adapter, 180 s per arm):** empty pools {PilotEmptyCtl} → {PilotEmptyLA}, coverage {PilotCovCtl} → {PilotCovLA}, pass@2 {PilotSolvedCtl} → {PilotSolvedLA}, {PilotLosses} losses, time ratio {PilotRatio}; iterative DFS matched the parent on {PilotEq}/{PilotEqN} undeadlined views at {PilotMsIter} vs {PilotMsParent} ms per step.
{CONF_BULLET}
![Figure 2](figures/clock.png)

![Figure 3](figures/pilot.png)

![Figure 4](figures/gallery.png)

## 5. Conclusion

A working test-time-trained ARC-AGI-2 solver loses most answers before selection begins. A one-line bound on sequential decoding explains a large part of this gap, the null results of several training changes, and a property of the benchmark: under a common budget split, more than a quarter of ARC-AGI-2 evaluation outputs cannot be decoded at all. Letting each output finish its first greedy path recovers candidates that equal slicing destroys, at no extra cost.

## Links

* Linked entry (L1): https://www.kaggle.com/code/huangjt66/arc2-paper-l1-length-aware-decoding
* Matched control (B0): https://www.kaggle.com/code/huangjt66/arc2-paper-b0-matched-control
* Code, notebooks and frozen records: https://github.com/huanji2tong/arc-prize-2026-out-of-time (MIT-0; upstream modules Apache-2.0), also attached as `arc-prize-2026-out-of-time.zip`
'''


def main():
    m = macros()
    conf = 'ConfN' in m
    m['CONF_SENTENCE'] = (f" On {m['ConfT']} untouched reserve tasks, pass@2 moves from {m['ConfSolvedCtl']} to {m['ConfSolvedLA']} of {m['ConfN']} outputs."
                          if conf else '')
    m['CONF_BULLET'] = (f"* **Reserve-task confirmation ({m['ConfT']} unseen tasks, {m['ConfN']} outputs):** pass@2 {m['ConfSolvedCtl']} → {m['ConfSolvedLA']} "
                        f"({m['ConfGains']} gains, {m['ConfLosses']} losses; 95% task-bootstrap CI [{m['ConfLo']}, {m['ConfHi']}] pp), "
                        f"empty pools {m['ConfEmptyCtl']} → {m['ConfEmptyLA']}, coverage {m['ConfCovCtl']} → {m['ConfCovLA']}.\n"
                        if conf else '')
    text = TEMPLATE
    for k, v in m.items():
        text = text.replace('{' + k + '}', str(v))
    left = re.findall(r'\{[A-Za-z_]+\}', text)
    assert not left, left
    (OUT / 'kaggle_writeup.md').write_text(text)
    # Writeup card image: Figure 1 at a 2:1 crop-free size.
    from PIL import Image
    img = Image.open(OUT / 'figures/teaser.png').convert('RGB')
    W, H = 1200, 600
    scale = min(W / img.width, H / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    card = Image.new('RGB', (W, H), 'white'); card.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    card.save(OUT / 'kaggle_card.png')
    print({'writeup': str(OUT / 'kaggle_writeup.md'), 'confirmation_included': conf, 'L1': m.get('ScoreLOne')})


if __name__ == '__main__':
    main()
