# -*- coding: utf-8 -*-
import unittest
import html
import re
import os
import sys
import importlib.util

def load_utils():
    module_name = 'karios.core.utils'
    file_path = os.path.join(os.getcwd(), 'karios', 'core', 'utils.py')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = load_utils()
sanitize_filename = utils.sanitize_filename

class TestSecurity(unittest.TestCase):
    def test_sanitize_filename(self):
        # Test logic from the actual karios.core.utils.py
        self.assertEqual(sanitize_filename("image; rm -rf /"), "image__rm_-rf__")
        self.assertEqual(sanitize_filename("CON.tif"), "safe_CON.tif")
        self.assertEqual(sanitize_filename("../../../etc/passwd"), "etc_passwd")
        self.assertEqual(sanitize_filename("....//etc/passwd"), "etc_passwd")
        self.assertEqual(sanitize_filename("../../../"), "unnamed")
        self.assertEqual(sanitize_filename("   "), "unnamed")
        self.assertEqual(sanitize_filename(""), "unnamed")

    def test_html_escaping_logic(self):
        malicious_input = "<script>alert('xss')</script>"
        escaped = html.escape(malicious_input)
        self.assertIn("&lt;script&gt;", escaped)

if __name__ == "__main__":
    unittest.main()
