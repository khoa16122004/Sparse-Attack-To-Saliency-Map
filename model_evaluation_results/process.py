import json
import os
import random


def select_one_path_per_class(data):
    selections = {}
    for class_name, paths in data.items():
        if not isinstance(paths, list) or len(paths) == 0:
            continue
        selections[class_name] = random.choice(paths)
    return selections


def main():
    random.seed(0)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for file_name in os.listdir(base_dir):
        if not file_name.endswith(".json"):
            continue
        if file_name.endswith("_selection.json"):
            continue

        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        selections = select_one_path_per_class(data)

        output_name = f"{os.path.splitext(file_name)[0]}_selection.json"
        output_path = os.path.join(base_dir, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selections, f, indent=2, ensure_ascii=False)

        print(f"Saved {output_name} with {len(selections)} selections")


if __name__ == "__main__":
    main()