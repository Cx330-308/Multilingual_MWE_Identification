from __future__ import annotations

import unittest

from mwe_roberta.cupt import CuptSentence, CuptToken


class PredictionSerializationTest(unittest.TestCase):
    def test_non_syntactic_rows_are_not_left_underspecified(self) -> None:
        sentence = CuptSentence(
            comments=["# text = du pain"],
            tokens=[
                CuptToken(["1-2", "du", "_", "_", "_", "_", "_", "_", "_", "_", "_"]),
                CuptToken(["1", "de", "_", "ADP", "_", "_", "2", "case", "_", "_", "_"]),
                CuptToken(["2", "le", "_", "DET", "_", "_", "3", "det", "_", "_", "_"]),
                CuptToken(["3", "pain", "_", "NOUN", "_", "_", "0", "root", "_", "_", "_"]),
                CuptToken(["3.1", "manger", "_", "VERB", "_", "_", "_", "_", "_", "_", "_"]),
            ],
        )

        predicted = sentence.with_predicted_mwes(["*", "*", "1:NID"])

        self.assertEqual(
            [token.mwe for token in predicted.tokens],
            ["*", "*", "*", "1:NID", "*"],
        )


if __name__ == "__main__":
    unittest.main()
