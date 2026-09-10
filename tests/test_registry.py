import unittest

from pytorch_yongli_skill.registry import DEFAULT_REGISTRY, SkillRegistry, SkillSpec


class TestSkillRegistry(unittest.TestCase):
    def test_register_and_lookup(self):
        registry = SkillRegistry()
        spec = SkillSpec(name="demo", description="demo skill")

        registry.register(spec)

        self.assertEqual(registry.get("demo"), spec)
        self.assertEqual(registry.names(), ["demo"])

    def test_default_registry_is_available(self):
        self.assertIsInstance(DEFAULT_REGISTRY, SkillRegistry)


if __name__ == "__main__":
    unittest.main()
