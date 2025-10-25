"""Identity data checks (pagoda.identity): name, other-model claims, wrong facts, held-out prompts, system prompt."""

from pagoda.identity import (conversation_problems, eval_set, fact_errors, name_count, near_duplicate,
                             other_model_claims, says_name, system_prompt, with_system_prompt)


def conv(cat, *turns):
    roles = ["user", "assistant"] * len(turns)
    return {"category": cat, "messages": [{"role": r, "content": t} for r, t in zip(roles, turns)]}


def test_says_name():
    assert says_name("I'm Pagoda-0.1, a small model.") and says_name("call me pagoda")
    assert not says_name("I'm a small language model.")
    assert name_count("Pagoda here. Pagoda-0.1, to be exact.") == 2


def test_other_model_claims():
    assert other_model_claims("Yes, I'm ChatGPT.")
    assert other_model_claims("I was created by OpenAI.")
    assert other_model_claims("I am based on Llama.")
    assert other_model_claims("As an AI developed by Google, I can help.")
    assert not other_model_claims("No, I'm not ChatGPT, and I wasn't made by OpenAI. I'm Pagoda-0.1.")
    assert not other_model_claims("I'm not based on GPT or any other model.")
    assert not other_model_claims("I'm much smaller than ChatGPT, which was made by OpenAI.")


def test_fact_errors():
    assert not fact_errors("I have about 100 million parameters and a context window of 1,024 tokens.")
    assert not fact_errors("The 5-day build has 110M parameters; I was trained by Zhen.")
    assert fact_errors("I have 1.5 billion parameters.")
    assert fact_errors("I was built by Meta.")
    assert fact_errors("I have a context window of 4,096 tokens.")


def test_held_out():
    ev = eval_set()
    assert len(ev["identity"]) == 20 and len(ev["unrelated"]) == 20
    assert near_duplicate("Is this ChatGPT??", "is this chatgpt")
    assert near_duplicate("Do you have a name? My daughter wants to know.", "Do you have a name?")
    assert not near_duplicate("Who are you?", "Who are you exactly?")
    assert near_duplicate("Hey Pagoda, a haiku about autumn leaves please?", "Write a haiku about autumn leaves.")
    assert not near_duplicate("Pagoda, write a haiku about the sea.", "Write a haiku about autumn leaves.")
    assert not near_duplicate("What's your name?", "What's the capital of France?")
    assert "near-duplicate of a held-out eval prompt" in conversation_problems(
        conv("name", "Is this ChatGPT?", "No, this is Pagoda-0.1."))


def test_category_rules():
    assert conversation_problems(conv("name", "Who are you?", "I'm Pagoda-0.1, a small language model.")) == []
    assert "never says the name" in conversation_problems(conv("denial", "Are you GPT-2?", "No."))
    assert "says the name in an ordinary task reply" in conversation_problems(
        conv("in_task", "Pagoda, give me a word for big.", "Huge. -- Pagoda"))
    assert conversation_problems(conv("in_task", "Pagoda, give me a word for big.", "Huge.")) == []
    assert conversation_problems({"category": "name", "messages": [{"role": "user", "content": "hi"}]})


def test_system_prompt_fraction_is_stable():
    msgs = [[{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": "ok"}] for i in range(4000)]
    first = [with_system_prompt(m, 0.25)[0]["content"] for m in msgs]
    assert first == [with_system_prompt(m, 0.25)[0]["content"] for m in msgs]
    frac = sum(f != m[0]["content"] for f, m in zip(first, msgs)) / len(msgs)
    assert 0.22 < frac < 0.28
    folded = with_system_prompt(msgs[0], 1.0)[0]["content"]
    assert folded.endswith("\n\nquestion 0") and "Pagoda" in folded
    assert with_system_prompt(msgs[0], 0) is msgs[0]
    assert "Pagoda-0.1" in system_prompt()
