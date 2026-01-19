from src.utils import JSONParser

VERDICT_MAPPING = {
    "true": True,
    "false": False,
    "True": True,
    "False": False,
    "TRUE": True,
    "FALSE": False,
    True: True,
    False: False,
}


class Evaluator:
    def __init__(self, data):
        self.types = data.keys()
        self.data = {}
        for samples in data.values():
            for sample in samples:
                self.data[sample["id"]] = sample

    def evaluate_json_result(
        self, llm_call_key_name="llm_call", verdict_key_name="verdict"
    ):
        results = {key: [] for key in self.types}
        for sample in self.data.values():
            try:
                json_value = JSONParser.extract_json_dict(sample[llm_call_key_name])
                sample["verdict"] = VERDICT_MAPPING[json_value[verdict_key_name]]
            except Exception as e:
                if "true" in sample[llm_call_key_name].lower():
                    sample["verdict"] = True
                else:
                    sample["verdict"] = False

            for type_ in sample["types"]:
                if type_ not in self.types:
                    continue
                results[type_].append(sample)
        accuracies = []
        for type_, samples in results.items():
            correct = 0
            for sample in samples:
                if sample["verdict"] == sample["Label"][0]:
                    correct += 1
            print(
                f"{type_}: Total {len(samples)} || Acc: {round(correct * 100/len(samples), 4)}"
            )
            accuracies.append(correct / len(samples))
        print(f"Total Acc: {round(sum(accuracies) * 100/len(accuracies), 4)}")
