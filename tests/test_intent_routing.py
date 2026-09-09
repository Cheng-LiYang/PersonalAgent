import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.agent.graph import KnowledgeAgent
from backend.agent.intent import route_intent
from backend.agent.calligraphy import CalligraphyIntent, analyze_calligraphy_intent
from backend.config import load_config
from backend.indexing import IndexService


class IntentTests(unittest.TestCase):
    def test_three_model_routes(self):
        for intent in ('chat', 'knowledge_base', 'calligraphy_lookup'):
            model = Mock()
            model.generate.return_value = json.dumps({'intent': intent})
            self.assertEqual(route_intent(model, '继续', [{'role': 'user', 'content': '查询文档'}]),
                             {'intent': intent, 'model_used': True})
            self.assertIn('查询文档', model.generate.call_args.args[1])

    def test_invalid_or_failed_router(self):
        for raw in ('[]', '{}', '{"intent":"invalid"}', 'not json'):
            model = Mock()
            model.generate.return_value = raw
            self.assertEqual(route_intent(model, '你是什么模型', [])['intent'], 'chat')
            self.assertEqual(route_intent(model, '查询上传的文档', [])['intent'], 'knowledge_base')
        model.generate.side_effect = RuntimeError('offline')
        self.assertEqual(route_intent(model, '查询春眠不觉晓的草书写法', [])['intent'], 'calligraphy_lookup')

    def test_preclassified_calligraphy_bypasses_keyword_gate(self):
        model = Mock()
        model.generate.return_value = json.dumps({'intent': 'calligraphy_lookup',
            'action': 'query_calligraphy_api', 'query_text': '春眠不觉晓'})
        result = analyze_calligraphy_intent(model, '春眠不觉晓', assume_intent=True)
        self.assertEqual(result.query_text, '春眠不觉晓')

    def test_dispatch_and_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config('config/docker.yaml')
            config['app']['data_dir'] = temp
            config['app']['knowledge_base'] = temp
            agent = KnowledgeAgent(config, IndexService(config).store)
            model = Mock()
            model.model = 'deepseek-chat'
            model.generate.side_effect = ['{"intent":"chat"}', '我是 deepseek-chat。']
            agent.nodes.llm = model
            with patch.object(agent.retriever, 'retrieve', create=True, side_effect=AssertionError('unexpected retrieval')):
                response = agent.ask('你是什么模型', 'chat-test')
            self.assertEqual(response.route, 'chat')
            self.assertFalse(response.requires_review)
            self.assertEqual(response.sources, [])
            self.assertEqual(len(agent.memory.history('chat-test')), 2)
            self.assertIn('deepseek-chat', model.generate.call_args.args[0])
            self.assertEqual(response.grounding['intent']['intent'], 'chat')
            self.assertEqual(agent.traces.get_run(response.run_id)['status'], 'completed')
            model.generate.side_effect = None
            model.generate.return_value = '{"intent":"knowledge_base"}'
            agent.graph = Mock()
            agent.graph.invoke.return_value = {'answer': '资料回答', 'route': 'multi_agent'}
            response = agent.ask('查询我的文档', 'kb-test')
            agent.graph.invoke.assert_called_once()
            self.assertEqual(response.answer, '资料回答')
            model.generate.return_value = '{"intent":"calligraphy_lookup"}'
            with patch('backend.agent.graph.analyze_calligraphy_intent', return_value=CalligraphyIntent(
                    'calligraphy_lookup', 'query_calligraphy_api', '春眠不觉晓')), \
                 patch.object(agent, '_lookup_calligraphy', return_value={
                    'answer': '草书结果', 'route': 'calligraphy_lookup'}) as lookup:
                response = agent.ask('展示春眠不觉晓的草书', 'ink-test')
                lookup.assert_called_once()
                self.assertEqual(response.route, 'calligraphy_lookup')


if __name__ == '__main__':
    unittest.main()
