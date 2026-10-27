# Examples and basis

Read when choosing between brevity and clarity, preserving a nuanced claim, or explaining the skill's rationale. Examples are hypothetical and demonstrate wording, not technical recommendations.

## Repairs

| Problem | Before | Clearer wording |
|---|---|---|
| Empty framing | It is important to note that the server restarted. | The server restarted. |
| Noun chain | Customer account access recovery workflow | Steps to restore access to a customer's account |
| Long interruption | The report, after three reviews and several revisions, was approved. | The report was approved after three reviews and several revisions. |
| Omitted grammatical cue | The report the editor revised is ready. | The report that the editor revised is ready. |
| Abstract action | An evaluation of the proposal was undertaken by the team. | The team evaluated the proposal. |
| Opaque metaphor | The function swallows the error. | The function ignores the error. |
| Ambiguous sequence | Backup complete delete old files. | If the backup succeeds, delete the old files. |

The right repair depends on context. A familiar technical term may be clearer than a long paraphrase, and a short embedded clause may already be effortless to read.

## Protected meaning

- Keep “The service may fail under heavy load.” Changing it to “The service fails under heavy load” removes uncertainty.
- Keep the exception in “Delete the logs after 30 days unless an investigation requires them.”
- Keep “should” in a recommendation. “Must” creates a different obligation.
- Keep both conditions in “Proceed only if the backup succeeds and the owner approves.” Do not replace “and” with “or.”
- Keep an attributed statement attributed. “The vendor says the tool is faster” does not establish that the tool is faster.
- Preserve source ambiguity when no defensible interpretation is available. Identify the ambiguity rather than inventing an actor, cause, or sequence.

## What this skill borrows

- **Caveman:** Direct answers, little filler, simple wording, exact technical details, and expansion when compression causes ambiguity. Prefer its lite approach, which keeps full sentences and articles. Do not adopt token savings as the primary objective or blanket removal of hedging.
- **Simple English:** Complete grammar, consistent terms, explicit actors, definitions suited to the reader, and conditions before commands. Do not adopt blanket modal substitutions, rigid sentence quotas, or a universal five-sentence response limit.
- **Unslop:** Identify concrete contextual defects before editing; preserve meaning and uncertainty; protect legitimate terminology; avoid edits based only on phrase matches or style scores.

These comparisons describe the local skill copies reviewed when Easyread was created on October 24, 2026. They are design influences, not runtime dependencies. Caveman's public repository: https://github.com/JuliusBrussee/caveman

## Research basis and limits

Clark, Martin, and Mahowald, “Why is AI-generated language so hard to understand?”, arXiv:2610.29162v1 (October 21, 2026): https://arxiv.org/html/2610.29162v1

The study reports that models often overuse linguistic features associated with processing difficulty, including unusual word combinations, long dependencies, stacked nouns, and omitted grammatical cues. Results vary by model and setting. An experiment with a small model shows that length penalties can encourage shorter grammatical alternatives, with some transfer beyond trained constructions.

The paper measures linguistic proxies rather than directly testing readers' comprehension of the sampled outputs. Translation normalizes language, complicating interpretation of that comparison. The training experiment demonstrates a possible mechanism, not the training cause of every model's behavior. The English-language findings do not establish universal rules for every language or genre.

Easyread combines those findings with editorial judgment. Its exact rule set has not been validated by this paper. Evaluate success through readers' understanding and ability to act, not token savings or a promise to make writing appear human.
