# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Materials dataset"""

import os
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
"""

_DESCRIPTION = """
"""

_ROOT = "/work/nihang/token_classification_ft_occupancy/datasets/ICSD_CN_oxide/"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "validation.txt"
_TEST_FILE = "test.txt"


class Materials(datasets.GeneratorBasedBuilder):
    """Materials dataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="materials", version=VERSION, description="Materials dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "-5",
                                "-4",
                                "-3",
                                "-2",
                                "-1",
                                "0",
                                "1",
                                "2",
                                "3",
                                "4",
                                "5",
                                "6",
                                "7",
                                "8",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        '''
        urls_to_download = {
            "train": f"{_ROOT}{_TRAINING_FILE}",
            "dev": f"{_ROOT}{_DEV_FILE}",
            "test": f"{_ROOT}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        '''
        
        data_files = {
            "train": os.path.join(_ROOT, _TRAINING_FILE),
            "validation": os.path.join(_ROOT, _DEV_FILE),
            "test": os.path.join(_ROOT, _TEST_FILE),
        }
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"], "split": "validation"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""

        #logger.info("? Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as f:

            guid = 0
            tokens = []
            ner_tags = []

            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())

            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
