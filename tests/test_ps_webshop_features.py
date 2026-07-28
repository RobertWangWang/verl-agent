"""Φ_webshop (W02/W04/W05) 测试: 提取器/解析/gold 往返 + WebshopEnvironmentManager 预测管线 (fake envs)。

运行: pytest tests/test_ps_webshop_features.py -v
"""

import re

from omegaconf import OmegaConf

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.verifiable_features import (
    create_webshop_feature_extractor,
    gold_predict_string,
    parse_predict_block,
    prediction_to_features,
    webshop_page_type,
    webshop_results_bin,
)

# --------------------------------------------------------------------------- #
# 页面脚本 (与 WebAgentTextEnv 渲染约定一致的 obs/available_actions 对)          #
# --------------------------------------------------------------------------- #

SEARCH_INFO = {'available_actions': {'has_search_bar': True, 'clickables': ['search']}}
RESULTS_OBS = "'Back to Search' [SEP] 'Page 1 (Total results: 50)' [SEP] 'Next >' [SEP] 'B001' [SEP] 'Red Shoe' [SEP] '$10.0'"
RESULTS_INFO = {'available_actions': {'has_search_bar': False, 'clickables': ['back to search', 'next >', 'b001']}}
ITEM_OBS = "'Back to Search' [SEP] '< Prev' [SEP] 'red shoe' [SEP] '$10.0' [SEP] 'Description' [SEP] 'Buy Now'"
ITEM_INFO = {'available_actions': {'has_search_bar': False, 'clickables': ['back to search', '< prev', 'description', 'features', 'reviews', 'buy now']}}
SUB_INFO = {'available_actions': {'has_search_bar': False, 'clickables': ['back to search', '< prev']}}


class TestPageType:
    def test_search_page(self):
        assert webshop_page_type('anything', SEARCH_INFO) == 'search'

    def test_results_page(self):
        assert webshop_page_type(RESULTS_OBS, RESULTS_INFO) == 'results'

    def test_item_page(self):
        assert webshop_page_type(ITEM_OBS, ITEM_INFO) == 'item'

    def test_item_sub_page(self):
        assert webshop_page_type("'description text'", SUB_INFO) == 'item_sub'

    def test_missing_info_defaults_results(self):
        assert webshop_page_type('x', {}) == 'results'


class TestResultsBin:
    def test_bins(self):
        assert webshop_results_bin("'Page 1 (Total results: 0)'") == '0'
        assert webshop_results_bin("'Page 1 (Total results: 7)'") == '1-10'
        assert webshop_results_bin(RESULTS_OBS) == '11-50'
        assert webshop_results_bin("'Page 1 (Total results: 500)'") == '50+'

    def test_absent_is_na(self):
        assert webshop_results_bin(ITEM_OBS) == 'na'
        assert webshop_results_bin('') == 'na'


class TestParsePredictBlockWebshop:
    def test_standard(self):
        parsed = parse_predict_block(
            '<predict>page_type: results; results_bin: 11-50; buy_now_visible: no</predict>')
        assert parsed == {'page_type': 'results', 'results_bin': '11-50', 'buy_now_visible': False}

    def test_aliases(self):
        parsed = parse_predict_block(
            '<predict>page_type: product page; results_bin: na; buy_now_visible: yes</predict>')
        assert parsed['page_type'] == 'item'
        assert parsed['results_bin'] == 'na'
        assert parsed['buy_now_visible'] is True

    def test_invalid_values_dropped(self):
        parsed = parse_predict_block(
            '<predict>page_type: checkout; results_bin: many; buy_now_visible: maybe</predict>')
        assert parsed is None  # 三个字段都无法归一 → 无可识别字段


class TestVerifyAndReward:
    def test_correct_prediction_full_reward(self):
        composite = create_webshop_feature_extractor()
        actual = composite.extract_all(RESULTS_OBS, [], RESULTS_INFO)
        parsed = parse_predict_block(
            '<predict>page_type: results; results_bin: 11-50; buy_now_visible: no</predict>')
        assert composite.compute_reward(prediction_to_features(parsed), actual) == 1.0

    def test_wrong_page_type_half_reward(self):
        composite = create_webshop_feature_extractor()
        actual = composite.extract_all(RESULTS_OBS, [], RESULTS_INFO)
        parsed = parse_predict_block(
            '<predict>page_type: item; results_bin: 11-50; buy_now_visible: no</predict>')
        assert composite.compute_reward(prediction_to_features(parsed), actual) == 0.5

    def test_probe_weight_zero_excluded(self):
        # buy_now_visible 默认权重 0: 单独预测它得 0 分 (不参与归一化)
        composite = create_webshop_feature_extractor()
        actual = composite.extract_all(ITEM_OBS, [], ITEM_INFO)
        parsed = parse_predict_block('<predict>buy_now_visible: yes</predict>')
        assert composite.compute_reward(prediction_to_features(parsed), actual) == 0.0


class TestGoldRoundTrip:
    def test_invariant_all_page_kinds(self):
        # S6 不变式: parse(gold) 对同一 actual_features 计 compute_reward 恒为 1.0
        composite = create_webshop_feature_extractor()
        for obs, info in [(RESULTS_OBS, RESULTS_INFO), (ITEM_OBS, ITEM_INFO),
                          ("'x'", SUB_INFO), ('start', SEARCH_INFO)]:
            actual = composite.extract_all(obs, [], info)
            gold = gold_predict_string(actual)
            parsed = parse_predict_block(f'<predict>{gold}</predict>')
            assert parsed is not None, gold
            assert composite.compute_reward(prediction_to_features(parsed), actual) == 1.0, gold


# --------------------------------------------------------------------------- #
# Manager 集成 (fake envs, 无需 WebShop 环境依赖)                               #
# --------------------------------------------------------------------------- #

RAW_RESET_OBS = 'Amazon Shopping Game [SEP] Instruction: [SEP] Find me a red shoe [SEP] Search'
RAW_RESULTS_OBS = ('Instruction: [SEP] Find me a red shoe [SEP] Back to Search [SEP] '
                   'Page 1 (Total results: 50) [SEP] Next > [SEP] B001 [SEP] Red Shoe [SEP] $10.0')


class FakeWebshopEnvs:
    """一步脚本化: search 页 → 搜索结果页 (Total results: 50)"""

    def __init__(self):
        self.num_envs = 1

    def reset(self):
        return [RAW_RESET_OBS], [dict(SEARCH_INFO, won=False)]

    def step(self, actions):
        infos = [dict(RESULTS_INFO, won=False, task_score=0.0)]
        return [RAW_RESULTS_OBS], [0.0], [False], infos


def webshop_projection_no_think(text_actions):
    actions, valids = [], []
    for text in text_actions:
        match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        actions.append(match.group(1).strip() if match else text[-20:])
        valids.append(1 if match else 0)
    return actions, valids


def make_config(enable, collect_gold=False):
    return OmegaConf.create({
        'env': {
            'env_name': 'Webshop',
            'history_length': 2,
            'webshop': {
                'use_small': True,
                'human_goals': False,
                'prediction': {
                    'enable': enable,
                    'horizon': 1,
                    'reward_mode': 'potential',
                    'collect_gold': collect_gold,
                    'feature_weights': None,
                },
            },
        },
    })


class TestWebshopManagerPrediction:
    def test_disabled_keeps_plain_flow(self):
        mgr = WebshopEnvironmentManager(FakeWebshopEnvs(), webshop_projection_no_think, make_config(False))
        obs, infos = mgr.reset({})
        assert '<predict>' not in obs['text'][0]
        _, _, _, infos = mgr.step(['<action>search[red shoe]</action>'])
        assert 'pred_accuracy' not in infos[0]

    def test_ps_prompt_contains_instruction(self):
        mgr = WebshopEnvironmentManager(FakeWebshopEnvs(), webshop_projection_no_think, make_config(True))
        obs, _ = mgr.reset({})
        assert 'page_type' in obs['text'][0] and '<predict>' in obs['text'][0]

    def test_correct_prediction_scores_one(self):
        mgr = WebshopEnvironmentManager(FakeWebshopEnvs(), webshop_projection_no_think, make_config(True))
        mgr.reset({})
        response = ('<predict>page_type: results; results_bin: 11-50; buy_now_visible: no</predict>'
                    '<action>search[red shoe]</action>')
        _, _, _, infos = mgr.step([response])
        assert infos[0]['pred_parse_valid'] is True
        assert infos[0]['pred_accuracy'] == 1.0

    def test_wrong_prediction_scores_partial(self):
        mgr = WebshopEnvironmentManager(FakeWebshopEnvs(), webshop_projection_no_think, make_config(True))
        mgr.reset({})
        response = ('<predict>page_type: item; results_bin: na; buy_now_visible: yes</predict>'
                    '<action>search[red shoe]</action>')
        _, _, _, infos = mgr.step([response])
        assert infos[0]['pred_accuracy'] == 0.0

    def test_missing_block_zero_and_invalid(self):
        mgr = WebshopEnvironmentManager(FakeWebshopEnvs(), webshop_projection_no_think, make_config(True))
        mgr.reset({})
        _, _, _, infos = mgr.step(['<action>search[red shoe]</action>'])
        assert infos[0]['pred_parse_valid'] is False
        assert infos[0]['pred_accuracy'] == 0.0

    def test_collect_gold_round_trip(self):
        mgr = WebshopEnvironmentManager(FakeWebshopEnvs(), webshop_projection_no_think, make_config(True, collect_gold=True))
        mgr.reset({})
        _, _, _, infos = mgr.step(['<action>search[red shoe]</action>'])
        gold = infos[0]['gold_predict']
        assert 'page_type: results' in gold and 'results_bin: 11-50' in gold
        parsed = parse_predict_block(f'<predict>{gold}</predict>')
        composite = create_webshop_feature_extractor()
        actual = composite.extract_all(mgr.pre_text_obs[0], [], dict(RESULTS_INFO))
        assert composite.compute_reward(prediction_to_features(parsed), actual) == 1.0
