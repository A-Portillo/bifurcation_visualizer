import unittest
from dynamics import parse_polynomial, equilibria, stability, critical_parameters

class DynamicsTests(unittest.TestCase):
    def test_classics(self):
        for source, rv, expected in [('r-x^2', 1, [-1, 1]), ('r*x-x^2', 2, [0, 2]), ('r*x-x^3', 1, [-1, 0, 1]), ('r*x+x^3', -1, [-1, 0, 1])]:
            roots = equilibria(parse_polynomial(source), rv)
            self.assertEqual(len(roots), len(expected))
            for actual, target in zip(roots, expected):
                self.assertAlmostEqual(actual, target)
        self.assertEqual(equilibria(parse_polynomial('r-x^2'), -1), [])

    def test_degenerate_stability(self):
        for source, expected in [('-x^3', 'Stable'), ('x^3', 'Unstable'), ('x^2', 'Semistable')]:
            self.assertEqual(stability(parse_polynomial(source), 0, 0), expected)
        self.assertIsNone(equilibria(parse_polynomial('r*x'), 0))
        self.assertEqual(equilibria(parse_polynomial('1'), 0), [])

    def test_safe_parser(self):
        for source in ['__import__("os")', 'x.__class__', 'sin(x)', 't+x', 'x/r', 'x^-1', 'x^21', 'True', 'x^2.5', '(x^20)^20']:
            with self.assertRaises(ValueError, msg=source):
                parse_polynomial(source)
        self.assertEqual(str(parse_polynomial('(r*x-x^3)/2')), 'r*x/2 - x**3/2')

    def test_critical_parameters(self):
        self.assertIn(0.0, critical_parameters(parse_polynomial('r-x^2'), -2, 2))
        self.assertIn(1.0, critical_parameters(parse_polynomial('(r-1)*x'), -2, 2))

    def test_streamlit_smoke(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file('app.py', default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        for name in ['Transcritical', 'Supercritical pitchfork', 'Subcritical pitchfork', 'Custom polynomial']:
            app.selectbox[0].select(name).run()
            self.assertEqual(len(app.exception), 0, name)
        app.text_area[0].set_value('r*x').run()
        self.assertEqual(len(app.exception), 0)
        for source in ['0', '1', '-x^3', 'x^2', '(r-1)*x+x^3-x^5']:
            app.text_area[0].set_value(source).run()
            self.assertEqual(len(app.exception), 0, source)
        app.text_area[0].set_value('t+x').run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(len(app.error))

if __name__ == '__main__':
    unittest.main()
