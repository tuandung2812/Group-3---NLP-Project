import os

def replace_html(old_file, new_file):
    with open(old_file, 'r', encoding='utf-8') as read_file, open(new_file, "w", encoding='utf-8') as write_file:
        for text in read_file:
            text = text.replace("&", "")
            text = text.replace("apos;", "'")
            text = text.replace("quot;", "\"")
            text = text.replace("#91;", "[")
            text = text.replace("#93;", "]")
            write_file.write(text)


if __name__ == "__main__":
    original_dir = "../data/original"
    processed_dir = "../data/processed"

    for sets in os.listdir(original_dir):
        dataset = os.path.join(original_dir , sets)
        new_dataset = os.path.join(processed_dir, sets)
        if not os.path.isdir(new_dataset):
            os.mkdir(new_dataset)
        for file_name in os.listdir(dataset):
            original_file = os.path.join(dataset,file_name)
            new_file = os.path.join(new_dataset, file_name)
            replace_html(original_file,new_file)

