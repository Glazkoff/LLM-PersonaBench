"""Prompt templates for every LLM operation an arm can request. One place, so the mutator sees
identical instructions regardless of which arm asks."""

GENOTYPE_SCHEMA = """Return ONLY a JSON object with exactly these keys:
{"role_definition": "<one paragraph: who the respondent is>",
 "trait_formulations": {"<trait>": "<one sentence describing this Big Five trait>", ...},
 "facet_formulations": {"<facet>": "<one sentence describing this facet>", ...},
 "critic_formulations": "<one paragraph: how to stay in character when answering>"}
Keep every trait and facet key exactly as in the input. Do not add keys. No commentary, no markdown."""

SYSTEM = ("You improve persona descriptions used to make a language model answer a personality "
          "questionnaire the way one specific person would. The persona is a JSON object. For each person, "
          "an intensity word ('very little', 'slightly', 'moderately', 'quite strongly', 'very strongly') is "
          "inserted automatically in front of every description according to that person's measured level, "
          "so each description must read correctly at every intensity. You only edit the wording.\n"
          + GENOTYPE_SCHEMA)

MUTATE = ("Rewrite this persona so the answering model becomes more sensitive to the person's actual trait "
          "levels while staying fluent. Change the wording, emphasis or structure of at least two entries.\n\n"
          "PERSONA:\n{genotype}")

CROSSOVER = ("Combine the two personas below into ONE child persona: keep the more precise formulation of each "
             "trait and facet, and write a role_definition that merges both.\n\nPARENT A:\n{a}\n\nPARENT B:\n{b}")

PARAPHRASE = ("Paraphrase every description in this persona, preserving meaning exactly. Do not add or remove "
              "information.\n\nPERSONA:\n{genotype}")

OPRO = ("Below are persona candidates with their scores (LOWER is better; the score is a probabilistic error on "
        "real people's held-out questionnaire answers). Study what the better ones do differently and propose "
        "ONE new persona you expect to score lower than all of them.\n\n{history}")

REFLECT = ("This persona was used to predict real people's questionnaire answers. Below is diagnostic feedback: "
           "the items where predictions were worst, with the direction of the error ('model too high' means the "
           "model predicted more of the trait than people reported). Reflect on what in the persona wording "
           "caused these errors, then return an improved persona.\n\nPERSONA:\n{genotype}\n\nFEEDBACK:\n{feedback}")

DE_STEP = ("Differential evolution step. Identify what differs between persona A and persona B (the 'difference "
           "vector'), apply that same kind of change to persona C, then cross the result with persona X by "
           "keeping X's entry wherever the changed entry is not clearly better. Return the final persona.\n\n"
           "A:\n{a}\n\nB:\n{b}\n\nC:\n{c}\n\nX:\n{x}")

META_MUTATE = ("You write instructions that other editors follow to improve personas. Here is a current "
               "instruction:\n\n{mutation_prompt}\n\nWrite a different instruction of similar length that would "
               "lead to more useful edits. Return ONLY the instruction text.")
