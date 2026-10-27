# Easyread

Easyread is an agent skill for writing, revising, and reviewing explanatory prose that people can understand on a first reading. It helps remove filler, unpack dense wording, and clarify how ideas connect while preserving facts and uncertainty.

The goal is to reduce the reader's effort. Sometimes that means cutting words; sometimes it means adding a word such as “that” or “because” to make a sentence easier to follow.

## Installation

```sh
npx skills add chand1012/skills --skill easyread
```

## How it works

Easyread adapts its edits to the intended audience and voice. It leads with the main point, removes filler, replaces unnecessary jargon, and keeps related words close together. It uses connecting words and explicit actors to make relationships and actions clear.

Meaning takes priority over brevity. Facts, conditions, exceptions, attribution, and uncertainty stay intact. For example, “The service may fail under heavy load” must keep “may”: removing it turns a possibility into a definite claim.

The skill supports three tasks:

- **Drafting:** Write an explanation from supplied notes.
- **Revision:** Return clearer wording while preserving the original meaning.
- **Review:** Identify specific problems and suggest repairs without rewriting the whole text.

Easyread suits explanations, documentation, and instructions. It keeps useful technical terms, explains them when the audience needs that context, and leaves clear text alone. It preserves code, identifiers, commands, quoted errors, and direct quotations unless you explicitly ask to edit them. It follows the requested style rather than imposing plain prose on fiction, poetry, or dialogue.

## Input and output examples

Three subagents generated the comparisons below, one for each category. Each wrote a baseline before reading [SKILL.md](SKILL.md), then read the skill and answered the same prompt with Easyread. The baseline prompts did not ask for awkward or overly complex writing. The outputs are reproduced below.

These are illustrative generations, not a controlled benchmark or evidence of a measured improvement. Wording varies between generations even without a skill. All scenarios are fictional.

### Technical documentation

This example asks for instructions with several conditions. Compare where each output puts the retry rules and how it explains the waiting periods.

**Input to both generations**

```text
Write technical documentation for junior developers in a neutral, practical
voice. Explain how to retry a fictional job API request.

POST /jobs accepts an Idempotency-Key header. Retry only HTTP 429 and 503,
at most three retries after the initial request. If Retry-After is present,
wait that many seconds; otherwise wait 1, 2, then 4 seconds. Reuse the same
key and request body on retries. Do not retry other statuses. A timeout
leaves the result unknown; check GET /jobs/by-key/{key} before deciding
whether to send another POST.

Keep all rules, and do not invent API behavior. Aim for 140–200 words.
```

**Output without Easyread**
```text
To retry POST /jobs safely, keep the original request body and the Idempotency-Key header value available for the entire operation. Every retry must use that same key and body.

Retry only when the response status is HTTP 429 or HTTP 503. Allow at most three retries after the initial request, for a maximum of four POST attempts. Do not retry any other response status.

Before each permitted retry, check the response for Retry-After. If it is present, wait the specified number of seconds. If it is absent, use these delays:

1. First retry: 1 second.
2. Second retry: 2 seconds.
3. Third retry: 4 seconds.

Handle timeouts separately from HTTP responses. A timeout leaves the request result unknown, so check GET /jobs/by-key/{key} before deciding whether to send another POST. Do not assume that the timeout means the job was not created. The lookup’s response behavior is not specified here, so do not infer a retry decision from an undocumented response.
```

**Output with Easyread**

```text
Retry `POST /jobs` only after an HTTP `429` or `503` response. You can retry at most three times after the initial request, for a maximum of four POST attempts. Do not retry other response statuses.

Keep the `Idempotency-Key` value and request body from the initial request. Reuse both on every retry.

Before each retry, choose the delay:

- If the response includes `Retry-After`, wait that many seconds.
- Otherwise, wait 1 second before the first retry, 2 seconds before the second, and 4 seconds before the third.

The fallback delay depends on which retry you are about to send. For example, if the second retry has no `Retry-After` value, wait 2 seconds before sending it.

If a POST request times out, its result is unknown. Check `GET /jobs/by-key/{key}` before deciding whether to send another POST, using the same key in the lookup path. The lookup’s response behavior is not specified here, so these rules alone do not tell you how to make that decision.
```
**What to notice:** The Easyread output leads with the retry condition and limit. It groups the two delay conditions together and adds a concrete example. Both outputs preserve the retry limit, key reuse, and uncertainty after a timeout. The Easyread output uses more words to explain the fallback delay.

### Professional emails

This example combines a request, a deadline, a possible consequence, and an optional call. Compare the subject lines and how each output tells the recipient what to do.

**Input to both generations**

```text
Write a professional email from Priya to client Jordan in a warm, direct
voice. The revised homepage designs are attached.

Feedback is needed by Tuesday, November 10 at 3 p.m. ET to keep the planned
Friday, November 13 handoff. Ask Jordan to choose option A or B and
consolidate team feedback in one reply. If feedback arrives later, the
handoff may move to the following week; this is not certain.

Offer a 15-minute call if useful, without proposing a time or implying a
meeting is booked. Include a subject line. Aim for 120–160 words.
```

**Output without Easyread**

```text
Subject: Homepage designs: feedback requested by November 10

Hi Jordan,

The revised homepage designs are attached for your review. Please choose
option A or B and send your team’s consolidated feedback in one reply by
Tuesday, November 10 at 3 p.m. ET. This will help us keep the planned
handoff on Friday, November 13.

As you review, it would be helpful to know which direction best supports
your priorities and whether there are any changes your team would like us
to make. Bringing everyone’s comments together will give us a clear set of
revisions to work from.

If feedback arrives after the deadline, the handoff may need to move to
the following week. If a conversation would help you decide, I’m happy to
arrange a 15-minute call.

Thanks,
Priya
```

**Output with Easyread**

```text
Subject: Please choose a homepage design by November 10

Hi Jordan,

The revised homepage designs are attached. Please choose option A or B and
send your team’s feedback in one reply by Tuesday, November 10 at 3 p.m. ET.
Receiving your feedback by then will help us keep the planned handoff on
Friday, November 13.

Please gather your team’s comments before replying so we can review them
together. In your reply, make clear which option you’ve chosen and which
changes you’d like us to make to that design.

If feedback arrives after the deadline, we may need to move the handoff to
the following week.

If it would be useful to talk through the two options, I’m happy to arrange
a 15-minute call.

Thanks,
Priya
```

**What to notice:** The Easyread subject names the requested decision. The body names feedback receipt as the condition for keeping the handoff and gives explicit instructions for gathering comments. Both outputs preserve the deadline, the possibility of a delay, and the optional call. The skill output still repeats parts of the request; applying a skill does not guarantee that every sentence needs no further editing.

## Research basis

Easyread draws on [“Why is AI-generated language so hard to understand?”](https://arxiv.org/html/2610.29162v1) by Thomas Hikaru Clark, Matias B. Martin, and Kyle Mahowald at New York University and the University of Texas at Austin (October 21, 2026).

The study found that models often overuse linguistic features associated with processing difficulty, including unusual word combinations, long dependencies between related words, stacked nouns, and omitted grammatical cues. Results varied across models and settings.

Easyread combines those findings with editorial judgment. The paper did not validate this skill's exact rules or directly test readers' comprehension of the sampled outputs. Applying the skill does not establish formal accessibility or controlled-language compliance; readability scores and word counts are diagnostic aids, not measures of success on their own.

See [SKILL.md](SKILL.md) for the full rules and [Examples and basis](references/examples-and-basis.md) for more examples, design influences, and research limits.
