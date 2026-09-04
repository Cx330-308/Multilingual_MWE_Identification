from math import isclose

from mwe_roberta.benchmark_analysis import _harmonic_f1, parse_detailed_results


def test_harmonic_f1_is_computed_after_macro_averaging():
    macro_precision = (0.8 + 0.2) / 2
    macro_recall = (0.2 + 0.8) / 2
    mean_per_language_f1 = (0.32 + 0.32) / 2

    official_macro_f1 = _harmonic_f1(macro_precision, macro_recall)

    assert isclose(official_macro_f1, 0.5)
    assert not isclose(official_macro_f1, mean_per_language_f1)


def test_parse_detailed_results_extracts_all_table_types():
    html = """
<h1>Cross-lingual macro-averages</h2>
<h2>General ranking:</h2>
<table><tbody><tr><td>17/17</td><td>62</td><td>60</td><td>61</td><td>70</td><td>64</td><td>67</td></tr></tbody></table>
<h2>Discontinuous vs Continuous VMWEs:</h2>
<table><tbody><tr><td>17/17</td><td>50</td><td>40</td><td>45</td><td>17/17</td><td>70</td><td>60</td><td>65</td></tr></tbody></table>
<h2>Unseen-in-train vs Seen-in-train VMWEs:</h2>
<table><tbody><tr><td>17/17</td><td>30</td><td>20</td><td>24</td><td>17/17</td><td>90</td><td>80</td><td>85</td></tr></tbody></table>
<h2>Variant-of-train vs Identical-to-train VMWEs:</h2>
<table><tbody><tr><td>17/17</td><td>60</td><td>50</td><td>55</td><td>17/17</td><td>90</td><td>90</td><td>90</td></tr></tbody></table>
<h2>Single-token vs Multi-token VMWEs:</h2>
<table><tbody><tr><td>15/15</td><td>20</td><td>40</td><td>27</td><td>17/17</td><td>70</td><td>60</td><td>65</td></tr></tbody></table>
<h1>Language-specific system ranking</h2>
<table><tbody><tr><td>XX</td><td>10</td><td>2</td><td>0.9</td><td>30</td><td>20</td><td>24</td><td>60</td><td>50</td><td>55</td><td>70</td><td>60</td><td>65</td></tr></tbody></table>
<h1>Results per VMWE category</h2>
<h2>XX</h2>
<table><tbody><tr><td>VID</td><td>60</td><td>50</td><td>55</td><td>70</td><td>60</td><td>65</td></tr></tbody></table>
"""

    results = parse_detailed_results(html)

    assert results.global_score["mwe_f1"] == 0.61
    assert len(results.phenomena) == 8
    assert results.phenomena[0]["group"] == "Discontinuous"
    assert results.languages[0]["language"] == "XX"
    assert results.languages[0]["unseen_f1"] == 0.24
    assert results.categories == [
        {
            "language": "XX",
            "category": "VID",
            "mwe_precision": 0.6,
            "mwe_recall": 0.5,
            "mwe_f1": 0.55,
            "token_precision": 0.7,
            "token_recall": 0.6,
            "token_f1": 0.65,
        }
    ]
