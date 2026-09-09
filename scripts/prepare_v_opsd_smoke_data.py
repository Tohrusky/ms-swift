#!/usr/bin/env python3
"""Convert copied Vision-OPD smoke data to the ms-swift OPSD schema."""

import argparse
import json
from pathlib import Path

PRIVILEGED_HINT = ('\n\nHere is a reference solution to this problem:\n'
                   '{solution}\n\n'
                   'After understanding the reference solution, please try to solve this '
                   'problem using your own approach below:')


def resolve_media_paths(paths: list[str], source_dir: Path) -> list[str]:
    resolved = []
    for path_str in paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = source_dir / path
        resolved.append(str(path.resolve()))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('smoke_training_data/data.jsonl'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('smoke_training_data/train.jsonl'),
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open(encoding='utf-8') as src, output_path.open('w', encoding='utf-8') as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row['question']
            solution = str(row['solution'])
            student_images = resolve_media_paths(row['images'], input_path.parent)
            teacher_images = resolve_media_paths(row['teacher_images'], input_path.parent)

            image_tags = question.count('<image>')
            if image_tags != len(student_images):
                raise ValueError(f"line {line_number}: question has {image_tags} <image> tags but "
                                 f"student has {len(student_images)} images")
            if image_tags != len(teacher_images):
                raise ValueError(f"line {line_number}: teacher prompt has {image_tags} <image> tags but "
                                 f"teacher has {len(teacher_images)} images")
            missing = [path for path in student_images + teacher_images if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"line {line_number}: missing media file: {missing[0]}")

            converted = {
                'messages': [{
                    'role': 'user',
                    'content': question
                }],
                'images': student_images,
                'teacher_prompt': question + PRIVILEGED_HINT.format(solution=solution),
                'teacher_images': teacher_images,
                'solution': solution,
            }
            dst.write(json.dumps(converted, ensure_ascii=False) + '\n')
            count += 1

    print(f"Wrote {count} OPSD samples to {output_path}")


if __name__ == '__main__':
    main()
