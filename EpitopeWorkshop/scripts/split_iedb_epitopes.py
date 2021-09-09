import logging
import os
from typing import Optional

import fire


class SplitIEDBEpitopes:
    def split_iebdb_file(self, path: str, files_amt: int = 100, target_dir: Optional[str] = None):
        target_dir = target_dir or os.path.join(os.path.dirname(path), 'split')
        base, ext = os.path.splitext(os.path.basename(path))
        lines_per_file = [[] for _ in range(files_amt)]
        logging.info("reading files")
        with open(path) as fp:
            batch = []
            for line_index, line in enumerate(fp):
                batch.append(line)
                if line_index % 2 == 1:
                    lines_per_file[(line_index // 2) % files_amt].extend(batch)
                    batch = []
        for index, file_lines in enumerate(lines_per_file):
            final_path = os.path.join(target_dir, f"{base}_{index}{ext}")
            logging.info(f"dumping to {final_path}")
            with open(final_path, 'w') as fp:
                fp.writelines(file_lines)
        return target_dir


if __name__ == '__main__':
    fire.Fire(SplitIEDBEpitopes)
