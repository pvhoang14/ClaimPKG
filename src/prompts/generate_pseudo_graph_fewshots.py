import copy
import random

FEWSHOTS_EXAMPLES = [
    "Claim: it has not 360 pages and the ISBN number is 1-56947-301-3.\nSubgraphs:\n<entity>1-56947-301-3</entity>||~isbn||unknown_0\nunknown_0||number of pages||<entity>360</entity>",
    "Claim: Well, Jason Sherlock did not have a nickname!\nSubgraphs:\n<entity>Jason Sherlock</entity>||nickname||unknown_0",
    "Claim: Do you know that Sweden is not the birthplace of Ace Wilder?\nSubgraphs:\n<entity>Ace Wilder</entity>||place of birth||<entity>Sweden</entity>",
    "Claim: Yes. Louis Herbert did not have a youth club.\nSubgraphs:\n<entity>Louis Herbert</entity>||youthclubs||unknown_0",
    "Claim: Yes, 4th Supply Battalion has a garrison.\nSubgraphs:\n<entity>4th Supply Battalion</entity>||garrison||unknown_0",
    "Claim: His name is Tommy Makem, he had a child.\nSubgraphs:\n<entity>Tommy Makem</entity>||children||unknown_0",
    "Claim: Garlic is the main ingredient of Ajoblanco, which is from Andalusia.\nSubgraphs:\n<entity>Ajoblanco</entity>||region||<entity>Andalusia</entity>\n<entity>Ajoblanco</entity>||ingredient||<entity>Garlic</entity>",
    "Claim: Akeem Priestley played for club RoPS and currently plays for the Orange County Blues FC, which is managed by Oliver Wyss.\nSubgraphs:\n<entity>Orange County Blues FC</entity>||manager||<entity>Oliver Wyss</entity>\n<entity>Orange County Blues FC</entity>||~clubs||<entity>Akeem Priestley</entity>\n<entity>Akeem Priestley</entity>||team||<entity>RoPS</entity>",
    "Claim: Arròs negre is a traditional dish from Spain, and from the Catalonia region, which is led by the Maria Norrfalk.\nSubgraphs:\n<entity>Arròs negre</entity>||country||<entity>Spain</entity>\n<entity>Arròs negre</entity>||region||<entity>Catalonia</entity>\n<entity>Catalonia</entity>||leader name||<entity>Maria Norrfalk</entity>",
    "Claim: It was founded in the Henderson, Nevada, currently still remaining in the Philippines.\nSubgraphs:\n<entity>Philippines</entity>||~location||unknown_0\nunknown_0||foundation place||<entity>Henderson, Nevada</entity>",
    "Claim: He is a Rhythm and Blues singer from Errata, Mississippi!\nSubgraphs:\n<entity>Rhythm and blues</entity>||~genre||unknown_0\nunknown_0||birth place||<entity>Errata, Mississippi</entity>\nunknown_0||background||unknown_1",
    "Claim: It's orbital period is 5.57 days and has the epoch date 27th June 2015.\nSubgraphs:\nunknown_0||epoch||<entity>2015-06-27</entity>\nunknown_0||~orbital period||unknown_1",
    "Claim: the county Thurleigh is Caernarfonshire.\nSubgraphs:\n<entity>Thurleigh</entity>||ceremonial county||<entity>Caernarfonshire</entity>",
    "Claim: Adam McQuaid's birthday is 0020!\nSubgraphs:\n<entity>Adam McQuaid</entity>||birth year||<entity>0020</entity>",
    "Claim: FOB Hop is a subgenre of alternative rock.\nSubgraphs:\n<entity>Alternative rock</entity>||music subgenre||<entity>FOB Hop</entity>",
]
PROMPT_TEMPLATE = """
### Task: Generate a reference graph to verify the following claim.
Only return the subgraphs following the format of provided examples and do NOT include other unnecessary information.

### Here are some examples:
{{example_text}}

### Claim: {{claim}}
Subgraphs:
""".strip()


def get_prompt_generated_pseudo_graphs_fewshots():
    example_list = copy.deepcopy(FEWSHOTS_EXAMPLES)
    random.shuffle(example_list)
    example_text = "\n\n".join(example_list)
    return PROMPT_TEMPLATE.replace("{{example_text}}", example_text)
