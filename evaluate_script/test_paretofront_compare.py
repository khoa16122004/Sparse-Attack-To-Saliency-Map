import math
import tempfile
import unittest
from pathlib import Path

from evaluate_script.paretofront_compare import _compare_fronts, _read_front


class DominanceTests(unittest.TestCase):
    def test_full_win_and_reverse(self):
        score = _compare_fronts([(0, 0)], [(1, 1), (2, 2)])
        self.assertEqual((score.ga_dominated, score.ga_count), (2, 2))
        self.assertEqual((score.coverage, score.reverse_coverage, score.advantage), (1, 0, 1))
        reverse = _compare_fronts([(1, 1), (2, 2)], [(0, 0)])
        self.assertEqual((reverse.coverage, reverse.reverse_coverage, reverse.advantage), (0, 1, -1))

    def test_equality_tradeoff_and_tied_coordinate(self):
        self.assertEqual(_compare_fronts([(1, 1)], [(1, 1)]).coverage, 0)
        self.assertEqual(_compare_fronts([(0, 2)], [(2, 0)]).coverage, 0)
        self.assertEqual(_compare_fronts([(1, 1)], [(1, 2)]).coverage, 1)
        self.assertEqual(_compare_fronts([(1, 1)], [(2, 1)]).coverage, 1)

    def test_unique_point_coverage_not_point_pair_count(self):
        score = _compare_fronts(
            [(0, 2), (2, 0), (0, 2)],
            [(3, 3), (3, 3), (1, 1)],
        )
        self.assertEqual((score.ga_count, score.nsgaii_count), (2, 2))
        self.assertEqual(score.ga_dominated, 1)
        self.assertEqual(score.coverage, 0.5)
        self.assertEqual(score.reverse_coverage, 0)

    def test_invalid_fronts_are_not_scored_as_wins(self):
        for ns, ga in [
            ([], [(0, 0)]), ([(0, 0)], []),
            ([(math.nan, 0)], [(0, 0)]), ([(0, 0)], [(math.inf, 0)]),
        ]:
            with self.subTest(ns=ns, ga=ga), self.assertRaises(ValueError):
                _compare_fronts(ns, ga)

    def test_reader_strict_validation_and_legacy_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "front.txt"
            path.write_text("\n1,2\n-1 3\n", encoding="utf-8")
            self.assertEqual(_read_front(path, strict=True), [(-1, 3), (1, 2)])
            for bad in ["bad 2", "nan 2", "1 inf", "1"]:
                with self.subTest(bad=bad):
                    path.write_text(f"0 0\n{bad}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, ":2:"):
                        _read_front(path, strict=True)
            path.write_text("header\n1 2\n", encoding="utf-8")
            self.assertEqual(_read_front(path), [(1, 2)])


if __name__ == "__main__":
    unittest.main()
