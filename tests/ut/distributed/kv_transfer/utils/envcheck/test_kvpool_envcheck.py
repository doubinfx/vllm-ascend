import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

ROOT = Path(__file__).resolve().parents[6] / "vllm_ascend/distributed/kv_transfer/utils/envcheck"
SPEC = importlib.util.spec_from_file_location("kvpool_envcheck", ROOT / "kvpool_envcheck.py")
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MEM_SCAN_SPEC = importlib.util.spec_from_file_location("mem_scan", ROOT / "artifacts/mem_scan.py")
assert MEM_SCAN_SPEC is not None
assert MEM_SCAN_SPEC.loader is not None
MEM_SCAN = importlib.util.module_from_spec(MEM_SCAN_SPEC)
sys.modules[MEM_SCAN_SPEC.name] = MEM_SCAN
MEM_SCAN_SPEC.loader.exec_module(MEM_SCAN)


class KvpoolEnvcheckTests(unittest.TestCase):
    catalog: dict[str, Any]

    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "rules.json").read_text(encoding="utf-8"))

    def test_tool_version_tracks_yuanrong_support(self):
        self.assertEqual(MODULE.TOOL_VERSION, "1.2.0")

    def rule(self, rule_id):
        return next(rule for rule in self.catalog["rules"] if rule["id"] == rule_id)

    def payload(self, results, *, fix=False, ssd_offload=False, ssd_path=None, actions=None, references=None):
        exit_code = MODULE.result_exit_code(results)
        arguments = {
            "hardware": "A3",
            "backend": "memcache",
            "managed_memory_mb": 32768,
            "deployment_mode": "multi-node",
            "link_type": "hccs",
            "ssd_offload": ssd_offload,
            "ssd_path": ssd_path,
            "peer_eid": None,
            "fix": fix,
            "output_dir": None,
            "catalog": str(ROOT / "rules.json"),
            "progress": "never",
        }
        actions = actions or []
        return {
            "tool_version": MODULE.TOOL_VERSION,
            "result_schema_version": MODULE.RESULT_SCHEMA_VERSION,
            "run_id": "20261015T120000Z_a1b2c3d4",
            "timestamp": 1792065600,
            "timestamp_iso": "2026-10-15T12:00:00+00:00",
            "invocation": {"arguments": arguments},
            "summary": MODULE.build_summary(results, exit_code),
            "repair": MODULE.build_repair_summary(fix, actions, MODULE.build_summary(results, exit_code)),
            "check_results": results,
            "reference_documents": references or [],
            "exit_code": exit_code,
        }

    def test_hdk_requirements_follow_hardware_backend_matrix(self):
        rule = self.rule("HDK-001")
        source = {"source": "/driver/version.info", "rc": 0, "matched_line": "Version=26.0.rc2"}
        cases = (
            ("A2", "memcache", "25.4.9", "BLOCK"),
            ("A2", "memcache", "25.5.0", "PASS"),
            ("A2", "mooncake", "25.5.0", "PASS"),
            ("A2", "yuanrong", "25.4.9", "BLOCK"),
            ("A2", "yuanrong", "25.5.0", "PASS"),
            ("A3", "memcache", "25.4.9", "BLOCK"),
            ("A3", "memcache", "25.5.0", "PASS"),
            ("A3", "memcache", "25.9.9", "PASS"),
            ("A3", "memcache", "26.0.rc2", "PASS"),
            ("A3", "mooncake", "25.4.9", "BLOCK"),
            ("A3", "mooncake", "25.5.0", "PASS"),
            ("A3", "mooncake", "26.0.0", "PASS"),
            ("A3", "yuanrong", "25.4.9", "BLOCK"),
            ("A3", "yuanrong", "25.5.0", "PASS"),
            ("A5", "memcache", "25.5.9", "BLOCK"),
            ("A5", "memcache", "25.6.0", "PASS"),
            ("A5", "mooncake", "25.6.rc3.1", "PASS"),
            ("A5", "yuanrong", "25.5.9", "BLOCK"),
            ("A5", "yuanrong", "25.6.0", "PASS"),
        )
        for hardware, backend, value, expected in cases:
            result = MODULE.evaluate_hdk_requirement(
                rule,
                {"hdk": {"hardware": hardware, "backend": backend, "version": value, "version_source": source}},
            )
            self.assertEqual(result["status"], expected)
        for requirement in rule["parameters"]["requirements"]:
            self.assertEqual(set(requirement), {"hardware", "backend", "minimum"})

    def test_raw_version_is_preserved(self):
        result = MODULE.evaluate_hdk_requirement(
            self.rule("HDK-001"),
            {
                "hdk": {
                    "hardware": "A3",
                    "backend": "memcache",
                    "version": "26.0.rc2",
                    "version_source": {"source": "npu-smi info", "rc": 0},
                }
            },
        )
        self.assertEqual(result["evidence"]["raw_version"], "26.0.rc2")
        self.assertEqual(result["evidence"]["normalized_version"], "26.0.0")
        self.assertIn("26.0.rc2", result["detail"])

    def test_hdk_unknown_version_is_blocking(self):
        result = MODULE.evaluate_hdk_requirement(
            self.rule("HDK-001"),
            {"hdk": {"hardware": "A3", "backend": "memcache", "version": None, "version_source": None}},
        )
        self.assertEqual(result["status"], "BLOCK")
        self.assertIn("Unknown evidence cannot be treated as a pass", result["detail"])

    def test_kernel_is_not_applicable_for_single_node(self):
        rule = self.rule("KERNEL-001")
        result = MODULE.evaluate_kernel_requirement(
            rule,
            {"kernel": {"deployment_mode": "single-node", "value": "5.10.0", "rc": 0, "stderr": ""}},
        )
        self.assertEqual(result["status"], "NOT_APPLICABLE")
        self.assertFalse(result["blocking"])
        self.assertIn("single-node", result["detail"])

    def test_kernel_blocks_old_version_only_for_multi_node(self):
        rule = self.rule("KERNEL-001")
        result = MODULE.evaluate_kernel_requirement(
            rule,
            {"kernel": {"deployment_mode": "multi-node", "value": "5.10.0", "rc": 0, "stderr": ""}},
        )
        self.assertEqual(result["status"], "BLOCK")
        self.assertTrue(result["blocking"])

    def test_a2_static_hugepages_use_free_two_mb_pages(self):
        meminfo = (
            "HugePages_Total:       60000\n"
            "HugePages_Free:        51200\n"
            "HugePages_Rsvd:            0\n"
            "Hugepagesize:          2048 kB\n"
        )
        with patch.object(MODULE.Path, "read_text", return_value=meminfo):
            result = MODULE.probe_a2_static_hugepages(102400)
        self.assertTrue(result["ok"])
        self.assertEqual(result["method"], "a2_static_hugepages")
        self.assertEqual(result["required_pages"], 51200)
        self.assertEqual(result["available_memory_mb"], 102400)
        self.assertEqual(result["reserved_pages"], 0)
        self.assertEqual(result["available_pages"], 51200)

        shortage = (
            "HugePages_Total:       60000\n"
            "HugePages_Free:        50000\n"
            "HugePages_Rsvd:            0\n"
            "Hugepagesize:          2048 kB\n"
        )
        with patch.object(MODULE.Path, "read_text", return_value=shortage):
            result = MODULE.probe_a2_static_hugepages(102400)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capacity")
        self.assertEqual(result["available_memory_mb"], 100000)

    def test_a2_static_hugepages_subtract_reserved_pages(self):
        meminfo = (
            "HugePages_Total:         100\n"
            "HugePages_Free:          100\n"
            "HugePages_Rsvd:           90\n"
            "Hugepagesize:          2048 kB\n"
        )
        with patch.object(MODULE.Path, "read_text", return_value=meminfo):
            result = MODULE.probe_a2_static_hugepages(100)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capacity")
        self.assertEqual(result["reserved_pages"], 90)
        self.assertEqual(result["available_pages"], 10)
        self.assertEqual(result["available_memory_mb"], 20)
        record = MODULE.evaluate_memory_scan(self.rule("MEM-001"), {"memory_scan": result})
        self.assertIn("reserved pages=90", record["detail"])
        self.assertIn("allocatable pages=10", record["detail"])

        all_reserved = meminfo.replace("HugePages_Rsvd:           90", "HugePages_Rsvd:          100")
        with patch.object(MODULE.Path, "read_text", return_value=all_reserved):
            result = MODULE.probe_a2_static_hugepages(2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capacity")
        self.assertEqual(result["available_pages"], 0)
        self.assertEqual(result["available_memory_mb"], 0)

    def test_a2_static_hugepages_require_complete_and_consistent_reservation_evidence(self):
        cases = {
            "missing": ("HugePages_Total:         100\nHugePages_Free:          100\nHugepagesize:          2048 kB\n"),
            "reserved_above_free": (
                "HugePages_Total:         100\n"
                "HugePages_Free:           20\n"
                "HugePages_Rsvd:           21\n"
                "Hugepagesize:          2048 kB\n"
            ),
            "free_above_total": (
                "HugePages_Total:          20\n"
                "HugePages_Free:          100\n"
                "HugePages_Rsvd:            0\n"
                "Hugepagesize:          2048 kB\n"
            ),
        }
        for name, meminfo in cases.items():
            with self.subTest(name=name), patch.object(MODULE.Path, "read_text", return_value=meminfo):
                result = MODULE.probe_a2_static_hugepages(2)
            self.assertFalse(result["ok"])
            self.assertEqual(result["failure_kind"], "capability")
            self.assertIsNone(result["available_pages"])
            self.assertIsNotNone(result["error"])

    def test_a2_static_hugepages_reject_non_two_mb_page_size(self):
        meminfo = """HugePages_Total:       60000\nHugePages_Free:        60000\nHugepagesize:          1048576 kB\n"""
        with patch.object(MODULE.Path, "read_text", return_value=meminfo):
            result = MODULE.probe_a2_static_hugepages(102400)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capability")

    def test_a5_memory_uses_mem_available(self):
        with patch.object(
            MODULE,
            "read_proc_meminfo",
            return_value={
                "MemTotal": "16777216 kB",
                "MemAvailable": "8388608 kB",
            },
        ):
            available = MODULE.probe_a5_memory_capacity(8192)
            insufficient = MODULE.probe_a5_memory_capacity(8193)
        self.assertTrue(available["ok"])
        self.assertEqual(available["available_memory_mb"], 8192)
        self.assertFalse(insufficient["ok"])
        self.assertEqual(insufficient["failure_kind"], "capacity")

    def test_a5_memory_fails_closed_without_mem_available(self):
        with patch.object(MODULE, "read_proc_meminfo", return_value={"MemTotal": "1024 kB"}):
            result = MODULE.probe_a5_memory_capacity(1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capability")

    def test_yuanrong_huge_tlb_uses_managed_memory_target(self):
        meminfo = {
            "HugePages_Total": "60000 kB",
            "HugePages_Free": "51200 kB",
            "HugePages_Rsvd": "0 kB",
            "Hugepagesize": "2048 kB",
        }
        with patch.object(MODULE, "read_proc_meminfo", return_value=meminfo):
            result = MODULE.probe_yuanrong_memory_capacity(102400)
        self.assertTrue(result["ok"])
        self.assertEqual(result["required_pages"], 51200)
        self.assertEqual(result["available_memory_mb"], 102400)
        record = MODULE.evaluate_yuanrong_memory(self.rule("YUANRONG-MEM-001"), {"yuanrong_memory": result})
        self.assertEqual(record["status"], "PASS")
        self.assertIn("target=102400MB", record["detail"])

    def test_yuanrong_huge_tlb_rejects_shortage_and_non_two_mb_pages(self):
        shortage = {
            "HugePages_Total": "60000 kB",
            "HugePages_Free": "50000 kB",
            "HugePages_Rsvd": "0 kB",
            "Hugepagesize": "2048 kB",
        }
        with patch.object(MODULE, "read_proc_meminfo", return_value=shortage):
            result = MODULE.probe_yuanrong_memory_capacity(102400)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capacity")
        record = MODULE.evaluate_yuanrong_memory(self.rule("YUANRONG-MEM-001"), {"yuanrong_memory": result})
        self.assertEqual(record["status"], "BLOCK")
        self.assertIn("reduce --managed-memory-mb", record["remediation"]["description"])

        non_two_mb = {**shortage, "Hugepagesize": "1048576 kB"}
        with patch.object(MODULE, "read_proc_meminfo", return_value=non_two_mb):
            result = MODULE.probe_yuanrong_memory_capacity(2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "capability")

    def test_yuanrong_runtime_requires_package_import_and_dscli(self):
        def passing_run(argv, timeout=8):
            if argv[0] == MODULE.sys.executable:
                return 0, "", ""
            if argv[0] == "/usr/bin/dscli":
                return 0, "dscli 0.4.0\n", ""
            raise AssertionError(argv)

        with (
            patch.object(MODULE.importlib.metadata, "version", return_value="0.4.0"),
            patch.object(MODULE.shutil, "which", return_value="/usr/bin/dscli"),
            patch.object(MODULE, "run_cmd", side_effect=passing_run),
        ):
            probe = MODULE.probe_yuanrong_runtime()
        self.assertTrue(probe["ok"])
        result = MODULE.evaluate_yuanrong_runtime(self.rule("DEP-YUANRONG-001"), {"yuanrong_runtime": probe})
        self.assertEqual(result["status"], "PASS")
        self.assertIn("version=0.4.0", result["detail"])

        failed = {
            "package": "openyuanrong-datasystem",
            "package_version": None,
            "import_rc": 1,
            "import_output": "No module named yr",
            "dscli": None,
            "dscli_rc": 127,
            "dscli_output": "dscli is not available",
            "ok": False,
        }
        result = MODULE.evaluate_yuanrong_runtime(self.rule("DEP-YUANRONG-001"), {"yuanrong_runtime": failed})
        self.assertEqual(result["status"], "BLOCK")
        self.assertIn("openyuanrong-datasystem is not installed", result["detail"])
        self.assertIn("dscli is not available", result["detail"])

    def test_a5_hugepages_warn_and_fix(self):
        rule = self.rule("A5-HUGEPAGE-001")
        before = {
            "method": "a5_hugepages",
            "ok": False,
            "evidence_complete": True,
            "entries": [
                {"node": 0, "page_size_kb": 2048, "configured_pages": 2, "path": "/tmp/hp2"},
                {"node": 0, "page_size_kb": 1048576, "configured_pages": 1, "path": "/tmp/hp1"},
            ],
            "residual_entries": [
                {"node": 0, "page_size_kb": 2048, "configured_pages": 2, "path": "/tmp/hp2"},
                {"node": 0, "page_size_kb": 1048576, "configured_pages": 1, "path": "/tmp/hp1"},
            ],
            "errors": [],
        }
        evaluated = MODULE.evaluate_a5_hugepages(rule, {"a5_hugepages": before})
        self.assertEqual(evaluated["status"], "WARN")
        self.assertFalse(MODULE.is_blocking_result(evaluated))
        self.assertEqual(evaluated["evidence"]["residual_entries"], before["residual_entries"])
        after = {
            "method": "a5_hugepages",
            "ok": True,
            "evidence_complete": True,
            "entries": [
                {"node": 0, "page_size_kb": 2048, "configured_pages": 0, "path": "/tmp/hp2"},
                {"node": 0, "page_size_kb": 1048576, "configured_pages": 0, "path": "/tmp/hp1"},
            ],
            "residual_entries": [],
            "errors": [],
        }
        with (
            patch.object(MODULE.os, "geteuid", return_value=0),
            patch.object(MODULE.Path, "write_text") as write_mock,
            patch.object(MODULE, "probe_a5_hugepages", return_value=after),
        ):
            final, action = MODULE.fix_a5_hugepages(
                rule,
                MODULE.CheckContext("A5", "mooncake", 1024, link_type="ub"),
                evaluated,
            )
        self.assertEqual(final["status"], "PASS")
        self.assertEqual(action["status"], "SUCCESS")
        self.assertEqual(write_mock.call_count, 2)

    def test_a5_hugepages_missing_evidence_blocks(self):
        probe = {
            "method": "a5_hugepages",
            "ok": False,
            "evidence_complete": False,
            "entries": [],
            "residual_entries": [],
            "errors": ["/sys/devices/system/node: unavailable"],
        }
        result = MODULE.evaluate_a5_hugepages(self.rule("A5-HUGEPAGE-001"), {"a5_hugepages": probe})
        self.assertEqual(result["status"], "BLOCK")
        self.assertTrue(MODULE.is_blocking_result(result))

    def test_memory_block_size_fails_closed_when_sysfs_is_unavailable_or_invalid(self):
        MEM_SCAN.get_memory_block_size.cache_clear()
        with (
            patch("builtins.open", side_effect=FileNotFoundError("missing")),
            self.assertRaisesRegex(RuntimeError, "block_size_bytes"),
        ):
            MEM_SCAN.get_memory_block_size()

        MEM_SCAN.get_memory_block_size.cache_clear()
        with (
            patch("builtins.open", mock_open(read_data="not-hex\n")),
            self.assertRaisesRegex(RuntimeError, "block_size_bytes"),
        ):
            MEM_SCAN.get_memory_block_size()

    def test_force_max_zoneorder_uses_python_kernel_release_and_fails_closed(self):
        MEM_SCAN.get_force_max_zoneorder.cache_clear()
        with (
            patch.object(MEM_SCAN.platform, "release", return_value="6.6.0"),
            patch("builtins.open", side_effect=FileNotFoundError("missing")),
        ):
            self.assertEqual(MEM_SCAN.get_force_max_zoneorder(), 11)

        MEM_SCAN.get_force_max_zoneorder.cache_clear()

        def read_ubuntu_os_release(path, *args, **kwargs):
            if path == "/etc/os-release":
                return io.StringIO("ID=ubuntu\n")
            raise FileNotFoundError(path)

        with (
            patch.object(MEM_SCAN.platform, "release", return_value=""),
            patch("builtins.open", side_effect=read_ubuntu_os_release),
        ):
            self.assertEqual(MEM_SCAN.get_force_max_zoneorder(), 13)

    def test_link_type_routes_rules(self):
        self.assertTrue(MODULE.applies(self.rule("HCCS-001"), "A3", "memcache", "hccs", "multi-node"))
        self.assertFalse(MODULE.applies(self.rule("HCCS-001"), "A3", "memcache", "roce", "multi-node"))
        self.assertTrue(MODULE.applies(self.rule("ROCE-001"), "A3", "memcache", "roce", "multi-node"))
        self.assertFalse(MODULE.applies(self.rule("ROCE-001"), "A3", "memcache", "hccs", "multi-node"))
        self.assertTrue(MODULE.applies(self.rule("LINGQU-001"), "A3", "mooncake", "hccs", "multi-node"))
        self.assertFalse(MODULE.applies(self.rule("LINGQU-001"), "A3", "mooncake", "hccs", "single-node"))
        self.assertFalse(MODULE.applies(self.rule("LINGQU-001"), "A3", "mooncake", "roce", "multi-node"))
        self.assertTrue(MODULE.applies(self.rule("A5-UB-001"), "A5", "mooncake", "ub", "multi-node"))
        self.assertFalse(MODULE.applies(self.rule("A5-UB-001"), "A5", "mooncake", "roce", "multi-node"))
        self.assertFalse(MODULE.applies(self.rule("ROCE-001"), "A5", "mooncake", "ub", "multi-node"))

    def test_a2_910b4_name_identifies_hardware_and_topology(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W)    Temp(C) |
| 0     910B4-1             | OK            | 100.0       42 |
| 0                         | 0000:C1:00.0  | 0           0  |
| 1     910B4-1             | OK            | 92.1        41 |
| 0                         | 0000:C2:00.0  | 0           0  |
| NPU     Chip              | Process id    | Process name |
"""
        topology = MODULE.parse_npu_topology(npu_output)
        self.assertEqual(topology["npu_ids"], [0, 1])
        self.assertEqual(topology["npu_count"], 2)
        self.assertEqual(topology["chips_per_npu"], 1)
        self.assertEqual(topology["device_names"], ["910B4-1"])
        with patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")):
            result = MODULE.probe_hardware(self.catalog["hardware_identification"])
        self.assertEqual(result["model"], "A2")
        self.assertEqual(result["match_details"]["A2"]["matched"], ["all_device_names"])
        self.assertEqual(result["observed"]["dies_by_npu"], {"0": 1, "1": 1})

    def test_mixed_910b_and_non_910b_names_are_not_a2(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W)    Temp(C) |
| 0     910B4-1             | OK            | 100.0       42 |
| 1     Ascend310P          | OK            | 100.0       42 |
| NPU     Chip              | Process id    | Process name |
"""
        with patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")):
            result = MODULE.probe_hardware(self.catalog["hardware_identification"])
        self.assertIsNone(result["model"])
        self.assertEqual(result["candidates"], [])

    def test_a3_name_and_two_dies_identify_partial_visibility(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W)    Temp(C) |
| Chip  Phy-ID              | Bus-Id        | AICore(%)  Memory-Usage(MB) |
| 4     Ascend910           | OK            | 100.0       42 |
| 0     8                   | 0000:8D:00.0  | 0           0 |
| 4     Ascend910           | OK            | -           42 |
| 1     9                   | 0000:8F:00.0  | 0           0 |
| 7     Ascend910           | OK            | 100.0       42 |
| 0     14                  | 0000:81:00.0  | 0           0 |
| 7     Ascend910           | OK            | -           42 |
| 1     15                  | 0000:83:00.0  | 0           0 |
| NPU     Chip              | Process id    | Process name |
"""
        topology = MODULE.parse_npu_topology(npu_output)
        self.assertEqual(topology["npu_ids"], [4, 7])
        self.assertEqual(topology["npu_count"], 2)
        self.assertEqual(topology["total_chip_count"], 4)
        self.assertEqual(topology["chips_per_npu"], 2)
        self.assertEqual(topology["device_names"], ["Ascend910"])
        self.assertEqual(topology["chip_ids_by_npu"], {4: [0, 1], 7: [0, 1]})
        self.assertEqual(topology["physical_ids_by_npu"], {4: [8, 9], 7: [14, 15]})
        with patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")):
            result = MODULE.probe_hardware(self.catalog["hardware_identification"])
        self.assertEqual(result["model"], "A3")
        self.assertEqual(result["match_details"]["A3"]["matched"], ["all_device_names", "dies_per_npu"])
        self.assertEqual(result["match_details"]["A3"]["dies_by_npu"], {4: 2, 7: 2})

    def test_a3_with_incomplete_die_visibility_is_rejected(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W)    Temp(C) |
| 4     Ascend910           | OK            | 100.0       42 |
| NPU     Chip              | Process id    | Process name |
"""
        with patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")):
            result = MODULE.probe_hardware(self.catalog["hardware_identification"])
        self.assertIsNone(result["model"])
        self.assertEqual(result["observed"]["dies_by_npu"], {"4": 1})

    def test_a5_950dt_identifies_a5(self):
        npu_output = """\
| NPU ID | Name             | Health        | Power(W)              |
| 0      | Ascend950DT      | OK            | 449.9                 |
| 1      | Ascend950DT      | OK            | 453.1                 |
| NPU ID | Process id       | Process name  | Process memory(MB)    |
"""
        topology = MODULE.parse_npu_topology(npu_output)
        self.assertEqual(topology["npu_ids"], [0, 1])
        self.assertEqual(topology["chips_per_npu"], 1)
        self.assertEqual(topology["device_names"], ["Ascend950DT"])
        with patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")):
            result = MODULE.probe_hardware(self.catalog["hardware_identification"])
        self.assertEqual(result["model"], "A5")
        self.assertEqual(result["match_details"]["A5"]["matched"], ["all_device_names", "dies_per_npu"])
        self.assertEqual(result["match_details"]["A5"]["dies_by_npu"], {0: 1, 1: 1})

    def test_a5_link_type_and_peer_eid_are_enforced(self):
        catalog = self.catalog
        detected = {
            "model": "A5",
            "candidates": ["A5"],
            "match_details": {},
            "sources": [],
        }
        with patch.object(MODULE, "probe_hardware", return_value=detected):
            self.assertEqual(
                MODULE.validate_inputs(catalog, "A5", "mooncake", "ub", "multi-node")[0],
                "A5",
            )
            with self.assertRaisesRegex(MODULE.InputError, "A5 supports only"):
                MODULE.validate_inputs(catalog, "A5", "mooncake", "roce", "multi-node")
        with self.assertRaisesRegex(MODULE.InputError, "A3 supports"):
            MODULE.validate_inputs(catalog, "A3", "mooncake", "ub", "multi-node")

    def test_hccs_state_and_topology_parsers(self):
        state_output = """\
 hccs health status : OK
 hccs lane mode : [4 4]
 hccs link lane list : [1111 1111]
 hccs link speed : [224 224]
 hccs retry count : [3 0]
 hccs error count : [0 0]
 hccs first error lane : NA
"""
        state = MODULE.parse_hccs_state(state_output)
        self.assertEqual(state["health_status"], "OK")
        self.assertEqual(state["lane_mode"], [4, 4])
        self.assertEqual(state["link_lane_list"], ["1111", "1111"])
        self.assertEqual(state["retry_count"], [3, 0])

        topo_output = """\
          Phy-ID0 Phy-ID1 Phy-ID2 Phy-ID3
Phy-ID0  X       SIO     HCCS_SW HCCS_SW
Phy-ID1  SIO     X       HCCS_SW HCCS_SW
Phy-ID2  HCCS_SW HCCS_SW X       SIO
Phy-ID3  HCCS_SW HCCS_SW SIO     X
"""
        topology = MODULE.parse_hccs_topology(topo_output)
        self.assertEqual(topology[0][1], "SIO")
        self.assertEqual(topology[0][2], "HCCS_SW")
        self.assertEqual(topology[3][2], "SIO")

    def test_hccs_probe_passes_static_and_protects_busy_devices(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W) |
| Chip  Phy-ID              | Bus-Id        | AICore(%) |
| 0     Ascend910           | OK            | 100.0 |
| 0     0                   | 0000:01:00.0  | 0 |
| 0     Ascend910           | OK            | - |
| 1     1                   | 0000:02:00.0  | 0 |
| 1     Ascend910           | OK            | 100.0 |
| 0     2                   | 0000:03:00.0  | 0 |
| 1     Ascend910           | OK            | - |
| 1     3                   | 0000:04:00.0  | 0 |
| NPU     Chip              | Process id    | Process name       | Process memory(MB) |
| 0       0                 | 1234          | python3            | 1024               |
"""
        topo_output = """\
          Phy-ID0 Phy-ID1 Phy-ID2 Phy-ID3
Phy-ID0  X       SIO     HCCS_SW HCCS_SW
Phy-ID1  SIO     X       HCCS_SW HCCS_SW
Phy-ID2  HCCS_SW HCCS_SW X       SIO
Phy-ID3  HCCS_SW HCCS_SW SIO     X
"""
        state = {
            "rc": 0,
            "health_status": "OK",
            "lane_mode": [4, 4, 4],
            "link_lane_list": ["1111", "1111", "1111"],
            "link_speed": [224, 224, 224],
            "retry_count": [0, 0, 0],
            "error_count": [0, 0, 0],
            "first_error_lane": "NA",
            "ok": True,
            "failures": [],
        }
        states = [{"npu_id": npu_id, "chip_id": chip_id, **state} for npu_id in (0, 1) for chip_id in (0, 1)]
        with (
            patch.object(MODULE, "run_cmd", side_effect=[(0, npu_output, ""), (0, topo_output, "")]),
            patch.object(MODULE, "query_hccs_states", return_value=(states, [], False)) as state_mock,
            patch.object(MODULE, "run_hccs_collective") as collective_mock,
        ):
            probe = MODULE.probe_hccs_diagnostics(
                "A3", "memcache", self.rule("HCCS-001")["parameters"], self.rule("HCCS-002")["parameters"]
            )
        self.assertTrue(probe["static"]["ok"])
        self.assertEqual(probe["static"]["topology_relation_counts"], {"SIO": 4, "HCCS_SW": 8})
        self.assertFalse(probe["dynamic"]["executed"])
        self.assertEqual(probe["dynamic"]["failure_kind"], "devices_busy")
        self.assertEqual(probe["dynamic"]["processes"][0]["pid"], 1234)
        state_mock.assert_called_once_with([(0, 0), (0, 1), (1, 0), (1, 1)], self.rule("HCCS-001")["parameters"])
        collective_mock.assert_not_called()

    def test_hccs_dynamic_static_failure_is_not_duplicate_blocker(self):
        probe = {
            "static": {"ok": False, "failure_kind": "link", "device_pairs": [], "error": "link failed"},
            "dynamic": {"ok": False, "executed": False, "failure_kind": "static_check_failed", "error": "not run"},
        }
        static = MODULE.evaluate_hccs_static(self.rule("HCCS-001"), {"hccs_diagnostics": probe})
        dynamic = MODULE.evaluate_hccs_dynamic(self.rule("HCCS-002"), {"hccs_diagnostics": probe})
        self.assertEqual(static["status"], "BLOCK")
        self.assertEqual(dynamic["status"], "NOT_EXECUTED")
        self.assertFalse(dynamic["blocking"])
        self.assertEqual(MODULE.build_summary([static, dynamic], 20)["blocking_count"], 1)

    def test_hccs_busy_skip_remains_blocking(self):
        probe = {
            "dynamic": {
                "ok": False,
                "executed": False,
                "failure_kind": "devices_busy",
                "processes": [{"pid": 1234}],
                "error": "NPU busy",
            }
        }
        result = MODULE.evaluate_hccs_dynamic(self.rule("HCCS-002"), {"hccs_diagnostics": probe})
        self.assertEqual(result["status"], "NOT_EXECUTED")
        self.assertTrue(result["blocking"])
        self.assertIn("Drain or migrate", result["remediation"]["description"])
        summary = MODULE.build_summary([result], 20)
        self.assertEqual(summary["blocking_not_executed_count"], 1)
        self.assertIn("not executed=1", MODULE.render_text(self.payload([result])))

    def test_hccs_probe_runs_dynamic_test_only_after_static_pass_and_idle(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W) |
| Chip  Phy-ID              | Bus-Id        | AICore(%) |
| 0     Ascend910           | OK            | 100.0 |
| 0     0                   | 0000:01:00.0  | 0 |
| 0     Ascend910           | OK            | - |
| 1     1                   | 0000:02:00.0  | 0 |
| NPU     Chip              | Process id    | Process name       | Process memory(MB) |
| No running processes found in NPU 0 |
"""
        topo_output = """\
          Phy-ID0 Phy-ID1
Phy-ID0  X       SIO
Phy-ID1  SIO     X
"""
        base = {
            "rc": 0,
            "health_status": "OK",
            "lane_mode": [4],
            "link_lane_list": ["1111"],
            "link_speed": [224],
            "retry_count": [5],
            "error_count": [1],
            "first_error_lane": "NA",
            "ok": True,
            "failures": [],
        }
        before = [{"npu_id": 0, "chip_id": chip_id, **base} for chip_id in (0, 1)]
        after = [{"npu_id": 0, "chip_id": chip_id, **base} for chip_id in (0, 1)]
        collective = {"ok": True, "rc": 0, "elapsed_seconds": 1.0, "result": {"ok": True}, "diagnostic": None}
        dynamic_parameters = {
            **self.rule("HCCS-002")["parameters"],
            "elements_per_peer": 4096,
            "iterations": 2,
            "timeout_seconds": 45,
        }
        with (
            patch.object(MODULE, "run_cmd", side_effect=[(0, npu_output, ""), (0, topo_output, "")]),
            patch.object(MODULE, "query_hccs_states", side_effect=[(before, [], False), (after, [], False)]),
            patch.object(MODULE, "run_hccs_collective", return_value=collective) as collective_mock,
        ):
            probe = MODULE.probe_hccs_diagnostics(
                "A3", "mooncake", self.rule("HCCS-001")["parameters"], dynamic_parameters
            )
        self.assertTrue(probe["static"]["ok"])
        self.assertTrue(probe["dynamic"]["ok"])
        self.assertTrue(probe["dynamic"]["executed"])
        self.assertEqual(probe["dynamic"]["error_counters"]["deltas"]["0:0"]["retry_count"], [0])
        collective_mock.assert_called_once_with(
            ROOT / "artifacts/hccs_test_a3.py", 2, elements_per_peer=4096, iterations=2, timeout_seconds=45
        )

    def test_hccs_counter_delta_uses_current_test_window(self):
        before = {"0:0": {"retry_count": [8, 2], "error_count": [1, 0]}}
        after = {"0:0": {"retry_count": [8, 3], "error_count": [1, 0]}}
        deltas, failures = MODULE.hccs_counter_deltas(before, after)
        self.assertEqual(deltas["0:0"]["retry_count"], [0, 1])
        self.assertEqual(deltas["0:0"]["error_count"], [0, 0])
        self.assertEqual(failures, ["card/die 0:0 link 2 retry_count increased by 1"])

    def test_hccs_interrupt_terminates_test_process_group(self):
        process = MagicMock()
        process.pid = 4321
        process.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with (
            patch.object(MODULE.subprocess, "Popen", return_value=process),
            patch.object(MODULE.os, "killpg") as kill_mock,
            self.assertRaises(KeyboardInterrupt),
        ):
            MODULE.run_hccs_collective(ROOT / "artifacts/hccs_test_a3.py", 2, 1024, 1, 10)
        kill_mock.assert_called_once_with(4321, MODULE.signal.SIGTERM)

    def test_a5_topology_and_port_parsers(self):
        topology_output = """\
NPU0 NPU1 NPU2 CPU Affinity
NPU0 X UB UB 288-383
NPU1 UB X UB 288-383
NPU2 UB UB X 96-191
"""
        topology = MODULE.parse_a5_topology(topology_output, "UB")
        self.assertTrue(topology["ok"])
        self.assertEqual(topology["npu_ids"], [0, 1, 2])
        self.assertEqual(topology["rows"]["1"], {0: "UB", 1: "X", 2: "UB"})

        port_output = """\
+---------+---------+---------------------+-----------+-------------+------------+
| UDie ID | Port ID | Speed Ability(Gbps) | Port Type | Link Status | Media Type |
+---------+---------+---------------------+-----------+-------------+------------+
| 1       | 0       | 400                 | UB        | UP          | Electrical |
| 1       | 8       | 200                 | ETH       | UP          | Optical    |
+---------+---------+---------------------+-----------+-------------+------------+
"""
        ports = MODULE.parse_a5_port_table(port_output)
        self.assertTrue(ports["ok"])
        self.assertEqual(ports["rows"][0]["port_type"], "UB")
        self.assertEqual(ports["rows"][1]["port_type"], "ETH")
        self.assertTrue(all(row["link_status"] == "UP" for row in ports["rows"]))

    def test_a5_topology_rejects_non_ub_relation(self):
        output = """\
NPU0 NPU1
NPU0 X SYS
NPU1 UB X
"""
        result = MODULE.parse_a5_topology(output, "UB")
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["mismatches"],
            [{"source": 0, "destination": 1, "relation": "SYS"}],
        )

    def test_a5_ub_probe_combines_topology_ports_urma_and_peer_ping(self):
        mapping = """\
0  0  0  0  Ascend950DT
1  1  0  1  Ascend950DT
"""
        topology = """\
NPU0 NPU1 CPU Affinity
NPU0 X UB 288-383
NPU1 UB X 288-383
"""
        ports = """\
| UDie ID | Port ID | Speed Ability(Gbps) | Port Type | Link Status | Media Type |
| 1       | 0       | 400                 | UB        | UP          | Electrical |
"""
        urma = """\
0  udmac0d1e2  UB  eid0 0000:0000:003f:0200:0010:0000:df28:0b04  ACTIVE
"""

        def fake_run(argv, timeout=8):
            if argv == ["npu-smi", "info", "-m"]:
                return 0, mapping, ""
            if argv == ["npu-smi", "info", "-t", "topo"]:
                return 0, topology, ""
            if argv[0] == "/usr/bin/hccn_tool":
                return 0, ports, ""
            if argv == ["/usr/bin/urma_admin", "show"]:
                return 0, urma, ""
            if argv == ["/usr/bin/urma_admin", "show", "topo"]:
                return 1, "", "Failed to recv netlink msg, ret:-1"
            if argv[0] == "urma_ping":
                return 0, "1 transmitted, 1 received, 0% packet loss", ""
            raise AssertionError(argv)

        context = MODULE.CheckContext(
            "A5",
            "mooncake",
            1024,
            deployment_mode="multi-node",
            link_type="ub",
            peer_eid="0000:0000:0040:0200:0010:0000:df28:1104",
        )
        rules = [
            self.rule("A5-UB-001"),
            self.rule("A5-URMA-001"),
            self.rule("A5-PING-001"),
        ]
        with (
            patch.object(MODULE, "run_cmd", side_effect=fake_run),
            patch.object(MODULE, "find_hccn_tool", return_value="/usr/bin/hccn_tool"),
            patch.object(
                MODULE.shutil,
                "which",
                side_effect=lambda name: f"/usr/bin/{name}" if name in ("urma_admin", "urma_ping") else None,
            ),
        ):
            result = MODULE.probe_a5_ub(context, rules)
        self.assertTrue(result["static"]["ok"])
        self.assertFalse(result["urma"]["ok"])
        self.assertTrue(result["peer_ping"]["ok"])
        self.assertEqual(
            MODULE.evaluate_a5_ub_static(self.rule("A5-UB-001"), {"a5_ub": result})["status"],
            "PASS",
        )
        self.assertEqual(
            MODULE.evaluate_a5_urma(self.rule("A5-URMA-001"), {"a5_ub": result})["status"],
            "BLOCK",
        )
        self.assertEqual(
            MODULE.evaluate_a5_peer_ping(self.rule("A5-PING-001"), {"a5_ub": result})["status"],
            "PASS",
        )

    def test_a5_urma_and_ping_parsers(self):
        endpoints = MODULE.parse_urma_endpoints(
            """\
0  udmac0d1e2  UB  eid0 0000:0000:003f:0200:0010:0000:df28:0b04  ACTIVE
1  udmac0d1e3  UB  eid0 0000:0000:003f:0300:0010:0000:df28:0b04  INACTIVE
"""
        )
        self.assertEqual(len(endpoints), 2)
        self.assertEqual(endpoints[0]["link"], "ACTIVE")
        ping = MODULE.parse_urma_ping("3 transmitted, 3 received, 0% packet loss, time 2000ms", 0)
        self.assertTrue(ping["ok"])
        self.assertEqual(ping["packet_loss_percent"], 0)

    def test_a5_secverify_evaluator_requires_documented_state(self):
        rule = self.rule("A5-SECVERIFY-001")
        passing = {
            "ok": True,
            "npu_ids": [0, 1],
            "states": [
                {"npu_id": 0, "enable": True, "mode": 0},
                {"npu_id": 1, "enable": True, "mode": 0},
            ],
            "errors": [],
        }
        self.assertEqual(
            MODULE.evaluate_a5_secverify(rule, {"a5_secverify": passing})["status"],
            "PASS",
        )
        passing["states"][1]["mode"] = 1
        result = MODULE.evaluate_a5_secverify(rule, {"a5_secverify": passing})
        self.assertEqual(result["status"], "BLOCK")
        self.assertIn("mismatches=1", result["detail"])

    def test_roce_ping_result_parses_real_success_output(self):
        output = """\
device 1 PING 10.30.126.23
recv seq=0,time=0.163000ms
recv seq=1,time=0.108000ms
recv seq=2,time=0.102000ms
3 packets transmitted, 3 received, 0.00% packet loss
"""
        result = MODULE.parse_roce_ping_result(output, 0)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["failure_kind"])
        self.assertEqual(result["transmitted"], 3)
        self.assertEqual(result["received"], 3)
        self.assertEqual(result["packet_loss_percent"], 0.0)

    def test_roce_ping_parameter_error_is_not_network_failure(self):
        output = "[ hccn_tool -i devid -ping ] needs the following parameters:"
        result = MODULE.parse_roce_ping_result(output, 4)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_kind"], "command")
        self.assertIn("invalid parameter", result["reason"])

    def test_roce_probe_covers_all_directed_pairs_with_near_mtu_packets(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W)    Temp(C) |
| 1     910B4-1             | OK            | 92.1        41 |
| 2     910B4-1             | OK            | 94.4        43 |
| NPU     Chip              | Process id    | Process name |
"""
        states = [
            {
                "npu_id": 1,
                "ip": "10.30.126.22",
                "netmask": "255.255.0.0",
                "link_status": "UP",
                "net_health": "Success",
                "mtu": 8192,
                "command_rc": {"ip": 0, "link": 0, "health": 0, "mtu": 0},
                "ok": True,
                "failures": [],
            },
            {
                "npu_id": 2,
                "ip": "10.30.126.23",
                "netmask": "255.255.0.0",
                "link_status": "UP",
                "net_health": "Success",
                "mtu": 8192,
                "command_rc": {"ip": 0, "link": 0, "health": 0, "mtu": 0},
                "ok": True,
                "failures": [],
            },
        ]
        counters = {"rc": 0, "counters": {key: 0 for key in MODULE.ROCE_ERROR_COUNTER_KEYS}, "diagnostic": None}
        ping_results = [
            {
                "source_npu": 1,
                "target_npu": 2,
                "target_ip": "10.30.126.23",
                "packet_size": 8160,
                "command": ["hccn_tool", "-i", "1", "-ping", "-g", "address", "10.30.126.23", "pkt", "8160"],
                "rc": 0,
                "output": "3 packets transmitted, 3 received, 0.00% packet loss",
                "stderr": "",
                "ok": True,
                "failure_kind": None,
                "reason": "ok",
                "transmitted": 3,
                "received": 3,
                "packet_loss_percent": 0.0,
            },
            {
                "source_npu": 2,
                "target_npu": 1,
                "target_ip": "10.30.126.22",
                "packet_size": 8160,
                "command": ["hccn_tool", "-i", "2", "-ping", "-g", "address", "10.30.126.22", "pkt", "8160"],
                "rc": 0,
                "output": "3 packets transmitted, 3 received, 0.00% packet loss",
                "stderr": "",
                "ok": True,
                "failure_kind": None,
                "reason": "ok",
                "transmitted": 3,
                "received": 3,
                "packet_loss_percent": 0.0,
            },
        ]
        with (
            patch.object(MODULE, "find_hccn_tool", return_value="/usr/local/Ascend/driver/tools/hccn_tool"),
            patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")),
            patch.object(MODULE, "probe_roce_npu_state", side_effect=states),
            patch.object(MODULE, "read_roce_error_counters", return_value=counters),
            patch.object(MODULE, "run_roce_ping", side_effect=ping_results) as ping_mock,
        ):
            local_parameters = {**self.rule("ROCE-002")["parameters"], "timeout_seconds": 9}
            result = MODULE.probe_roce_diagnostics(
                "A2", "memcache", self.rule("ROCE-001")["parameters"], local_parameters
            )
        self.assertTrue(result["link"]["ok"])
        self.assertTrue(result["local"]["ok"])
        self.assertNotIn("error_counters", result["link"])
        self.assertIn("error_counters", result["local"])
        self.assertEqual(result["local"]["packet_size"], 8160)
        self.assertEqual(result["local"]["direction_count"], 2)
        self.assertEqual(ping_mock.call_count, 2)
        ping_mock.assert_any_call("/usr/local/Ascend/driver/tools/hccn_tool", 1, 2, "10.30.126.23", 8160, 9, 0.0)
        ping_mock.assert_any_call("/usr/local/Ascend/driver/tools/hccn_tool", 2, 1, "10.30.126.22", 8160, 9, 0.0)

    def test_roce_static_failure_skips_dynamic_check_without_duplicate_blocker(self):
        npu_output = """\
| NPU   Name                | Health        | Power(W)    Temp(C) |
| 1     910B4-1             | OK            | 92.1        41 |
| 2     910B4-1             | OK            | 94.4        43 |
| NPU     Chip              | Process id    | Process name |
"""
        states = [
            {
                "npu_id": 1,
                "ip": "10.30.126.22",
                "netmask": "255.255.0.0",
                "link_status": "DOWN",
                "net_health": "Success",
                "mtu": 8192,
                "command_rc": {"ip": 0, "link": 0, "health": 0, "mtu": 0},
                "ok": False,
                "failures": ["Link=DOWN"],
            },
            {
                "npu_id": 2,
                "ip": "10.30.126.23",
                "netmask": "255.255.0.0",
                "link_status": "UP",
                "net_health": "Success",
                "mtu": 8192,
                "command_rc": {"ip": 0, "link": 0, "health": 0, "mtu": 0},
                "ok": True,
                "failures": [],
            },
        ]
        with (
            patch.object(MODULE, "find_hccn_tool", return_value="/usr/local/Ascend/driver/tools/hccn_tool"),
            patch.object(MODULE, "run_cmd", return_value=(0, npu_output, "")),
            patch.object(MODULE, "probe_roce_npu_state", side_effect=states),
            patch.object(MODULE, "read_roce_error_counters") as counter_mock,
            patch.object(MODULE, "run_roce_ping") as ping_mock,
        ):
            probe = MODULE.probe_roce_diagnostics(
                "A2", "memcache", self.rule("ROCE-001")["parameters"], self.rule("ROCE-002")["parameters"]
            )
        self.assertFalse(probe["link"]["ok"])
        self.assertFalse(probe["local"]["executed"])
        counter_mock.assert_not_called()
        ping_mock.assert_not_called()

        link = MODULE.evaluate_roce_link(self.rule("ROCE-001"), {"roce_diagnostics": probe})
        local = MODULE.evaluate_roce_local(self.rule("ROCE-002"), {"roce_diagnostics": probe})
        self.assertEqual(link["status"], "BLOCK")
        self.assertTrue(link["blocking"])
        self.assertEqual(local["status"], "NOT_EXECUTED")
        self.assertFalse(local["blocking"])
        self.assertEqual(MODULE.result_exit_code([link, local]), 20)

    def test_roce_counter_delta_belongs_to_dynamic_data_path(self):
        probe = {
            "hardware": "A2",
            "backend": "memcache",
            "available_npu_ids": [0, 1],
            "npu_states": [],
            "link": {"ok": True, "failure_kind": None, "error": None},
            "local": {
                "ok": False,
                "executed": True,
                "failure_kind": "data_path",
                "packet_size": 8160,
                "direction_count": 2,
                "directions": [],
                "error_counters": {"before": {}, "after": {}, "deltas": {"0": {"mac_rx_fcs_err_pkt_num": 1}}},
                "error": "NPU 0 mac_rx_fcs_err_pkt_num increased by 1",
            },
        }
        link = MODULE.evaluate_roce_link(self.rule("ROCE-001"), {"roce_diagnostics": probe})
        local = MODULE.evaluate_roce_local(self.rule("ROCE-002"), {"roce_diagnostics": probe})
        self.assertEqual(link["status"], "PASS")
        self.assertNotIn("error_counters", link["evidence"])
        self.assertEqual(local["status"], "BLOCK")
        self.assertEqual(local["evidence"]["error_counters"]["deltas"]["0"]["mac_rx_fcs_err_pkt_num"], 1)
        self.assertIn("new error counters=1", local["detail"])

    def test_roce_records_expose_clear_positioning(self):
        for rule_id in ("ROCE-001", "ROCE-002"):
            positioning = self.rule(rule_id)["positioning"]
            self.assertEqual(set(positioning), {"layer", "target", "coverage", "excludes"})
            self.assertTrue(positioning["excludes"])
        record = MODULE.make_record(self.rule("ROCE-001"), "PASS", {}, "test")
        report = MODULE.render_text(self.payload([record]))
        for label in ("Layer:", "Target:", "Coverage:", "Excludes:"):
            self.assertIn(label, report)

    def test_roce_ping_command_includes_near_mtu_packet_size(self):
        output = "device 1 PING 10.30.126.23\n3 packets transmitted, 3 received, 0.00% packet loss"
        with patch.object(MODULE, "run_cmd", return_value=(0, output, "")) as command_mock:
            result = MODULE.run_roce_ping("/driver/hccn_tool", 1, 2, "10.30.126.23", 8160)
        self.assertTrue(result["ok"])
        command_mock.assert_called_once_with(
            ["/driver/hccn_tool", "-i", "1", "-ping", "-g", "address", "10.30.126.23", "pkt", "8160"], timeout=15
        )

    def test_result_record_schema(self):
        result = MODULE.make_record(self.rule("DOCKER-001"), "PASS", {}, "test")
        self.assertEqual(
            set(result),
            {
                "id",
                "status",
                "problem",
                "impact",
                "detail",
                "evidence_method",
                "evidence",
                "applicability",
                "blocking",
                "remediation",
            },
        )

    def test_docker_prereqs_accepts_hccn_conf_or_hccn_tool(self):
        rule = self.rule("DOCKER-001")
        base = {
            "docker": True,
            "npu_smi": True,
            "davinci_device_count": 8,
            "driver_dir": True,
            "devices": ["/dev/davinci0"],
        }
        for hccn_conf, hccn_tool, expected in (
            (True, False, "PASS"),
            (False, True, "PASS"),
            (True, True, "PASS"),
            (False, False, "BLOCK"),
        ):
            result = MODULE.evaluate_docker_prereqs(
                rule,
                {"docker_prereqs": {**base, "hccn_conf": hccn_conf, "hccn_tool": hccn_tool}},
            )
            self.assertEqual(result["status"], expected)
        tool_fallback = MODULE.evaluate_docker_prereqs(
            rule,
            {"docker_prereqs": {**base, "hccn_conf": False, "hccn_tool": True}},
        )
        self.assertEqual(tool_fallback["evidence"]["satisfied_hccn_entry"], "hccn_tool")
        self.assertIn("HCCN entry point=hccn_tool", tool_fallback["detail"])

    def test_a5_docker_prereqs_require_complete_ub_environment(self):
        rule = self.rule("A5-DOCKER-001")
        probe = {
            "docker": True,
            "npu_smi": True,
            "hccn_tool": True,
            "urma_admin": True,
            "urma_ping": True,
            "devices": ["/dev/davinci0"],
            "davinci_device_count": 8,
            "davinci_manager": True,
            "hisi_hdc": True,
            "driver_dir": True,
            "ummu_device": True,
            "uburma_device": True,
            "hccn_conf": True,
            "hccl_rootinfo": True,
            "hixlep_dir": True,
            "route_conf": True,
        }
        passing = MODULE.evaluate_docker_prereqs(rule, {"a5_docker_prereqs": probe})
        self.assertEqual(passing["status"], "PASS")
        probe["hccn_tool"] = False
        failing = MODULE.evaluate_docker_prereqs(rule, {"a5_docker_prereqs": probe})
        self.assertEqual(failing["status"], "BLOCK")
        self.assertIn("hccn_tool", failing["detail"])
        for field in ("davinci_manager", "hisi_hdc", "route_conf"):
            with self.subTest(field=field):
                missing = {**probe, "hccn_tool": True, field: False}
                result = MODULE.evaluate_docker_prereqs(rule, {"a5_docker_prereqs": missing})
                self.assertEqual(result["status"], "BLOCK")
                self.assertIn(field, result["detail"])

    def test_catalog_contains_only_current_scenarios_and_docker_schema(self):
        rule_ids = {rule["id"] for rule in self.catalog["rules"]}
        self.assertEqual(
            rule_ids,
            {
                "MEM-001",
                "MEM-A5-001",
                "YUANRONG-MEM-001",
                "A5-HUGEPAGE-001",
                "CANN-001",
                "HDK-001",
                "A5-DOCKER-001",
                "A5-SECVERIFY-001",
                "A5-UB-001",
                "A5-URMA-001",
                "A5-PING-001",
                "KERNEL-001",
                "LINGQU-001",
                "HCCS-001",
                "HCCS-002",
                "ROCE-001",
                "ROCE-002",
                "DOCKER-001",
                "DEP-YUANRONG-001",
                "DEP-MOONCAKE-001",
                "DEP-MOONCAKE-002",
                "DEP-MEMCACHE-001",
                "DEP-MEMCACHE-002",
                "SSD-001",
            },
        )
        self.assertEqual(self.catalog["schema_version"], "7.1")
        self.assertTrue(self.rule("A5-HUGEPAGE-001")["blocking"])
        self.assertEqual(set(self.rule("DOCKER-001")["parameters"]), {"required_all", "required_any"})

    def test_memory_evidence_is_compact(self):
        probe = {
            "ok": False,
            "script": "mem_scan.py",
            "scan_scope": "a3_fixed_ranges",
            "managed_memory_mb": 32768,
            "failure_kind": "capacity",
            "available_memory_mb": 8192.0,
            "matched_granularity_mb": None,
            "scans": {
                "1024": {"rc": 0, "total_available_mb": 4096.0},
                "2": {"rc": 0, "total_available_mb": 8192.0},
            },
            "error": None,
        }
        result = MODULE.evaluate_memory_scan(self.rule("MEM-001"), {"memory_scan": probe})
        self.assertEqual(result["status"], "BLOCK")
        self.assertEqual(result["evidence"], probe)
        self.assertIn("shortage 24576MB", result["remediation"]["description"])

    def test_memory_scan_falls_back_to_2mb_and_checks_global_capacity(self):
        primary = {
            "min_segment_mb": 1024,
            "rc": 0,
            "total_available_mb": 8000.0,
            "numa_summary": [{"node": 0, "available_mb": 8000.0, "elapsed": "1s"}],
            "elapsed": "1s",
            "diagnostic": None,
        }
        fallback = {
            "min_segment_mb": 2,
            "rc": 0,
            "total_available_mb": 40000.0,
            "numa_summary": [
                {"node": 0, "available_mb": 39000.0, "elapsed": "1s"},
                {"node": 1, "available_mb": 1000.0, "elapsed": "1s"},
            ],
            "elapsed": "1s",
            "diagnostic": None,
        }
        with (
            patch.object(MODULE.Path, "exists", return_value=True),
            patch.object(MODULE, "run_memory_scan", side_effect=[primary, fallback]) as scan_mock,
        ):
            result = MODULE.probe_a3_memory_scan(32768)
        self.assertTrue(result["ok"])
        self.assertEqual(result["matched_granularity_mb"], 2)
        self.assertEqual(result["available_memory_mb"], 40000.0)
        self.assertEqual([call.args[1] for call in scan_mock.call_args_list], [1024, 2])

    def test_text_does_not_repeat_remediation(self):
        rule = self.rule("DOCKER-001")
        passed = MODULE.make_record(rule, "PASS", {}, "test")
        blocked = MODULE.make_record(rule, "BLOCK", {}, "test")
        results = [passed, blocked]
        payload = self.payload(results)
        rendered = MODULE.render_text(payload)
        advice = blocked["remediation"]["description"]
        self.assertEqual(rendered.count(advice), 1)
        self.assertIn("Conclusion: Issue not detected", rendered)
        self.assertIn("Conclusion: Issue detected", rendered)
        self.assertNotIn("Check item:", rendered)
        self.assertEqual(rendered.count("Scenario:"), 2)

    def test_text_records_effective_inputs(self):
        result = MODULE.make_record(self.rule("DOCKER-001"), "PASS", {}, "test")
        without_ssd = MODULE.render_text(self.payload([result]))
        self.assertEqual(
            without_ssd.splitlines()[0],
            f"KV Pool Environment Check Report v{MODULE.TOOL_VERSION} | run_id=20261015T120000Z_a1b2c3d4",
        )
        for expected in (
            "- hardware: A3",
            "- backend: memcache",
            "- managed_memory_mb: 32768",
            "- ssd_offload: false",
            "- ssd_path: null",
            "- fix: false",
            "- output_dir: null",
            f"- catalog: {ROOT / 'rules.json'}",
            "- progress: never",
        ):
            self.assertIn(expected, without_ssd)

        with_ssd = MODULE.render_text(self.payload([result], fix=True, ssd_offload=True, ssd_path="/dev/nvme1n1"))
        for expected in ("- ssd_offload: true", "- ssd_path: /dev/nvme1n1", "- fix: true"):
            self.assertIn(expected, with_ssd)

    def test_reference_document_has_no_redundant_annotation(self):
        self.assertEqual(
            set(MODULE.REFERENCE_DOCUMENTS[0]),
            {"title", "url", "source_commit", "reviewed_at"},
        )
        result = MODULE.make_record(self.rule("DOCKER-001"), "PASS", {}, "test")
        payload = self.payload([result], references=MODULE.REFERENCE_DOCUMENTS)
        reference_line = MODULE.render_text(payload).split("References:\n", 1)[1].splitlines()[0]
        self.assertEqual(
            reference_line, f"- {MODULE.REFERENCE_DOCUMENTS[0]['title']}: {MODULE.REFERENCE_DOCUMENTS[0]['url']}"
        )

    def test_text_conclusion_covers_all_statuses(self):
        rule = self.rule("DOCKER-001")
        status_map = {
            "PASS": "Issue not detected",
            "BLOCK": "Issue detected",
            "WARN": "Risk detected; confirmation required",
            "MANUAL_REQUIRED": "Manual confirmation required",
            "NOT_EXECUTED": "Not executed in the current environment",
            "NOT_APPLICABLE": "Not applicable to the current deployment mode",
            "UNKNOWN": "Cannot confirm",
        }
        for status, conclusion in status_map.items():
            result = MODULE.make_record(rule, status, {}, "test")
            results = [result]
            payload = self.payload(results)
            self.assertIn(f"Conclusion: {conclusion}", MODULE.render_text(payload))

    def test_cli_has_only_output_dir_contract(self):
        parser = MODULE.build_parser(self.catalog)
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertIn("--output-dir", options)
        self.assertIn("--fix", options)
        self.assertIn("--link-type", options)
        self.assertIn("--deployment-mode", options)
        self.assertIn("--sync-panorama", options)
        self.assertIn("--progress", options)
        self.assertNotIn("--format", options)
        self.assertNotIn("--output", options)
        self.assertNotIn("--authorize", options)
        self.assertNotIn("--export-panorama", options)
        help_text = parser.format_help()
        for expected in (
            "Check arguments (required in check mode)",
            "SSD Offload (optional)",
            "Output arguments (optional)",
            "Repair flow (optional)",
            "Rule management (no check arguments required)",
            "--managed-memory-mb",
            "example: 32768",
            "Authorize and execute supported local repairs",
            "Validate the rule file",
            "Exit codes: 0=passed",
            "Progress output: auto=interactive terminal only",
        ):
            self.assertIn(expected, help_text)

    def test_progress_reporter_emits_only_start_and_end(self):
        stream = io.StringIO()
        reporter = MODULE.ProgressReporter(True, stream=stream)
        with reporter.stage("slow check", 1, 2):
            time.sleep(0.01)
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("[progress 1/2] slow check: started", lines[0])
        self.assertIn("[progress 1/2] slow check: finished, elapsed", lines[1])
        self.assertNotIn("running", stream.getvalue())

    def test_progress_reporter_disabled_is_silent(self):
        stream = io.StringIO()
        reporter = MODULE.ProgressReporter(False, stream=stream)
        with reporter.stage("silent check"):
            time.sleep(0.001)
        self.assertEqual(stream.getvalue(), "")

    def test_cli_supported_ranges_come_from_catalog(self):
        catalog = json.loads(json.dumps(self.catalog))
        help_text = MODULE.build_parser(catalog).format_help()
        self.assertIn("--hardware {A2,A3,A5}", help_text)
        self.assertIn("--backend {memcache,mooncake,yuanrong}", help_text)
        self.assertIn("--link-type {roce,hccs,ub}", help_text)
        self.assertIn("--peer-eid EID", help_text)

    def test_yuanrong_rejects_ssd_offload(self):
        stderr = io.StringIO()
        with (
            patch.object(MODULE, "is_docker_runtime", return_value=True),
            patch("sys.stderr", stderr),
        ):
            rc = MODULE.main(
                [
                    "--hardware",
                    "A3",
                    "--backend",
                    "yuanrong",
                    "--managed-memory-mb",
                    "32768",
                    "--deployment-mode",
                    "single-node",
                    "--link-type",
                    "hccs",
                    "--ssd-offload",
                ]
            )
        self.assertEqual(rc, 2)
        self.assertIn("--ssd-offload is not supported for --backend yuanrong", stderr.getvalue())

    def test_peer_eid_requires_a5_multi_node_ub(self):
        stderr = io.StringIO()
        with (
            patch.object(MODULE, "is_docker_runtime", return_value=True),
            patch("sys.stderr", stderr),
        ):
            rc = MODULE.main(
                [
                    "--hardware",
                    "A3",
                    "--backend",
                    "mooncake",
                    "--managed-memory-mb",
                    "1024",
                    "--deployment-mode",
                    "multi-node",
                    "--link-type",
                    "hccs",
                    "--peer-eid",
                    "0000:0000:0000:0000:0000:0000:0000:0001",
                ]
            )
        self.assertEqual(rc, 2)
        self.assertIn("--peer-eid requires --hardware A5", stderr.getvalue())

    def test_probe_plan_follows_rule_order(self):
        context = MODULE.CheckContext("A3", "memcache", 32768, link_type="hccs")
        rules = [
            rule
            for rule in self.catalog["rules"]
            if MODULE.applies(rule, context.hardware, context.backend, context.link_type, context.deployment_mode)
            and MODULE.feature_applies(rule, context.ssd_offload)
        ]
        self.assertEqual(
            list(MODULE.build_probe_plan(rules)),
            [
                "memory_scan",
                "hdk",
                "kernel",
                "manual_confirmation",
                "hccs_diagnostics",
                "docker_prereqs",
                "package_versions",
            ],
        )

    def test_a5_rule_and_probe_plan(self):
        context = MODULE.CheckContext(
            "A5",
            "mooncake",
            32768,
            deployment_mode="multi-node",
            link_type="ub",
        )
        rules = [
            rule
            for rule in self.catalog["rules"]
            if MODULE.applies(
                rule,
                context.hardware,
                context.backend,
                context.link_type,
                context.deployment_mode,
            )
            and MODULE.feature_applies(rule, context.ssd_offload)
        ]
        self.assertEqual(
            {rule["id"] for rule in rules},
            {
                "MEM-A5-001",
                "A5-HUGEPAGE-001",
                "CANN-001",
                "HDK-001",
                "A5-DOCKER-001",
                "A5-UB-001",
                "A5-URMA-001",
                "A5-PING-001",
                "DEP-MOONCAKE-001",
                "DEP-MOONCAKE-002",
            },
        )
        self.assertEqual(
            list(MODULE.build_probe_plan(rules)),
            [
                "a5_memory",
                "a5_hugepages",
                "hdk",
                "a5_docker_prereqs",
                "a5_ub",
                "package_versions",
                "glibc",
            ],
        )
        memcache_context = MODULE.CheckContext(
            "A5",
            "memcache",
            32768,
            deployment_mode="single-node",
            link_type="ub",
        )
        memcache_ids = {
            rule["id"]
            for rule in self.catalog["rules"]
            if MODULE.applies(
                rule,
                memcache_context.hardware,
                memcache_context.backend,
                memcache_context.link_type,
                memcache_context.deployment_mode,
            )
            and MODULE.feature_applies(rule, False)
        }
        self.assertIn("A5-SECVERIFY-001", memcache_ids)
        self.assertIn("DEP-MEMCACHE-001", memcache_ids)
        self.assertIn("DEP-MEMCACHE-002", memcache_ids)
        self.assertNotIn("A5-PING-001", memcache_ids)

    def test_yuanrong_rule_applicability_across_hardware(self):
        cases = {
            ("A2", "roce", "multi-node"): {
                "YUANRONG-MEM-001",
                "CANN-001",
                "HDK-001",
                "ROCE-001",
                "ROCE-002",
                "DOCKER-001",
                "DEP-YUANRONG-001",
            },
            ("A3", "hccs", "multi-node"): {
                "YUANRONG-MEM-001",
                "CANN-001",
                "HDK-001",
                "HCCS-001",
                "HCCS-002",
                "DOCKER-001",
                "DEP-YUANRONG-001",
            },
            ("A5", "ub", "multi-node"): {
                "YUANRONG-MEM-001",
                "CANN-001",
                "HDK-001",
                "A5-DOCKER-001",
                "A5-UB-001",
                "A5-URMA-001",
                "A5-PING-001",
                "DEP-YUANRONG-001",
            },
        }
        for (hardware, link_type, deployment_mode), expected in cases.items():
            with self.subTest(hardware=hardware, link_type=link_type):
                rule_ids = {
                    rule["id"]
                    for rule in self.catalog["rules"]
                    if MODULE.applies(rule, hardware, "yuanrong", link_type, deployment_mode)
                    and MODULE.feature_applies(rule, False)
                }
                self.assertTrue(expected <= rule_ids)
                self.assertNotIn("A5-SECVERIFY-001", rule_ids)
                self.assertNotIn("A5-HUGEPAGE-001", rule_ids)
                self.assertNotIn("SSD-001", rule_ids)
                self.assertNotIn("DEP-MOONCAKE-001", rule_ids)
                self.assertNotIn("DEP-MEMCACHE-001", rule_ids)

    def test_package_probe_uses_names_from_applicable_rules(self):
        rules = [
            {"evaluator": "package_version", "parameters": {"package": "package-a"}},
            {"evaluator": "package_version", "parameters": {"package": "package-b"}},
        ]
        with patch.object(
            MODULE.importlib.metadata, "version", side_effect=lambda name: f"{name}-version"
        ) as version_mock:
            result = MODULE.PROBES["package_versions"](MODULE.CheckContext("A3", "memcache", 32768), rules)
        self.assertEqual(result["versions"], {"package-a": "package-a-version", "package-b": "package-b-version"})
        self.assertEqual([call.args[0] for call in version_mock.call_args_list], ["package-a", "package-b"])

    def test_ssd_unknown_backend_does_not_fall_back_to_mooncake(self):
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE.probe_ssd_storage(directory, "yuanrong")
        self.assertFalse(result["ok"])
        self.assertIn("No SSD Offload storage check strategy is implemented for backend yuanrong", result["error"])

    def test_memcache_ssd_rejects_block_device_with_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            device = Path(directory) / "ssd"
            device.touch()
            lsblk = json.dumps(
                {
                    "blockdevices": [
                        {
                            "path": str(device),
                            "type": "disk",
                            "fstype": None,
                            "mountpoints": [None],
                            "pkname": None,
                        },
                        {
                            "path": f"{device}1",
                            "type": "part",
                            "fstype": None,
                            "mountpoints": [None],
                            "pkname": device.name,
                        },
                    ]
                }
            )
            with (
                patch.object(MODULE.Path, "is_block_device", return_value=True),
                patch.object(MODULE, "run_cmd", return_value=(0, lsblk, "")),
            ):
                result = MODULE.probe_ssd_storage(str(device), "memcache")
        self.assertFalse(result["ok"])
        self.assertEqual(result["partition_count"], 1)
        self.assertIn("partitions", result["error"])

    def test_memcache_packages_are_required_for_normal_and_ssd_modes(self):
        normal_ids = {
            rule["id"]
            for rule in self.catalog["rules"]
            if MODULE.applies(rule, "A3", "memcache", "roce", "multi-node") and MODULE.feature_applies(rule, False)
        }
        ssd_ids = {
            rule["id"]
            for rule in self.catalog["rules"]
            if MODULE.applies(rule, "A3", "memcache", "roce", "multi-node") and MODULE.feature_applies(rule, True)
        }
        self.assertIn("DEP-MEMCACHE-002", normal_ids)
        self.assertIn("DEP-MEMCACHE-002", ssd_ids)
        self.assertEqual(self.rule("DEP-MEMCACHE-001")["parameters"]["threshold"], "1.3.0")
        self.assertEqual(self.rule("DEP-MEMCACHE-002")["parameters"]["threshold"], "1.3.0")
        blocked = MODULE.evaluate_package_version(
            self.rule("DEP-MEMCACHE-002"),
            {"package_versions": {"versions": {"memcache-hybrid": "1.2.9"}}},
        )
        passed = MODULE.evaluate_package_version(
            self.rule("DEP-MEMCACHE-002"),
            {"package_versions": {"versions": {"memcache-hybrid": "1.3.0"}}},
        )
        self.assertEqual(blocked["status"], "BLOCK")
        self.assertEqual(passed["status"], "PASS")

    def test_a3_mooncake_package_threshold_depends_on_hdk(self):
        rule = self.rule("DEP-MOONCAKE-001")
        package_versions = {
            "package_versions": {"versions": {"mooncake-transfer-engine-npu": "0.3.10"}},
        }

        old_hdk = MODULE.evaluate_package_version(
            rule,
            {
                **package_versions,
                "hdk": {"hardware": "A3", "backend": "mooncake", "version": "25.5.0"},
            },
        )
        self.assertEqual(old_hdk["status"], "BLOCK")
        self.assertEqual(old_hdk["evidence"]["required_version"], "0.3.11.post1")

        new_hdk = MODULE.evaluate_package_version(
            rule,
            {
                **package_versions,
                "hdk": {"hardware": "A3", "backend": "mooncake", "version": "26.0.0"},
            },
        )
        self.assertEqual(new_hdk["status"], "PASS")
        self.assertIsNone(new_hdk["evidence"]["required_version"])
        self.assertEqual(new_hdk["evidence"]["threshold_exception"]["minimum_hdk"], "26.0.0")

        for hardware in ("A2", "A5"):
            with self.subTest(hardware=hardware):
                unchanged = MODULE.evaluate_package_version(
                    rule,
                    {
                        **package_versions,
                        "hdk": {"hardware": hardware, "backend": "mooncake", "version": "26.0.0"},
                    },
                )
                self.assertEqual(unchanged["status"], "BLOCK")
                self.assertEqual(unchanged["evidence"]["required_version"], "0.3.11.post1")

    def test_output_bundle_uses_one_run_id_for_result_and_report(self):
        result = MODULE.make_record(self.rule("DOCKER-001"), "PASS", {}, "test")
        payload = self.payload([result])
        with tempfile.TemporaryDirectory() as directory:
            result_path, report_path = MODULE.write_output_bundle(Path(directory), payload)
            self.assertEqual(result_path.name, "kvpool_envcheck_20261015T120000Z_a1b2c3d4.result.json")
            self.assertEqual(report_path.name, "kvpool_envcheck_20261015T120000Z_a1b2c3d4.report.txt")
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["run_id"], payload["run_id"])
            self.assertEqual(saved["invocation"]["arguments"]["fix"], False)
            for removed_top_level_input in (
                "hardware",
                "backend",
                "managed_memory_mb",
                "ssd_offload",
                "ssd_path",
                "actions",
            ):
                self.assertNotIn(removed_top_level_input, saved)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(f"run_id={payload['run_id']}", report)
            self.assertIn(f"Complete result (JSON): {result_path}", report)
            completion = MODULE.render_completion(payload, result_path, report_path)
            self.assertEqual(
                completion,
                "\n".join(
                    [
                        "Check complete",
                        "Final result: PASS",
                        f"run_id: {payload['run_id']}",
                        f"Complete result (JSON): {result_path}",
                        f"Report (TEXT): {report_path}",
                        "",
                    ]
                ),
            )

    def test_fix_is_authorization_and_panorama_sync_is_explicit(self):
        blocked = MODULE.evaluate_manual_confirmation(self.rule("LINGQU-001"), {"manual_confirmation": {}})
        context = MODULE.CheckContext("A3", "memcache", 32768)
        final_results, actions = MODULE.execute_fix(self.catalog, context, [blocked])
        self.assertEqual(final_results, [blocked])
        self.assertEqual(
            actions,
            [
                {
                    "id": "LINGQU-001",
                    "type": "manual_switch_upgrade",
                    "status": "MANUAL",
                    "message": blocked["remediation"]["description"],
                }
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            design = Path(directory) / "DESIGN.md"
            design.write_text("# Design\n", encoding="utf-8")
            MODULE.write_panorama(design, self.catalog)
            content = design.read_text(encoding="utf-8")
            self.assertIn("<!-- BEGIN GENERATED RULE PANORAMA -->", content)
            self.assertIn("| MEM-001 |", content)
            self.assertIn("<!-- END GENERATED RULE PANORAMA -->", content)

    def test_memory_fix_drops_caches_and_rescans(self):
        before_probe = {
            "method": "a3_mem_scan",
            "ok": False,
            "script": "mem_scan.py",
            "scan_scope": "a3_fixed_ranges",
            "managed_memory_mb": 32768,
            "failure_kind": "capacity",
            "available_memory_mb": 0.0,
            "matched_granularity_mb": None,
            "scans": {},
            "error": None,
        }
        after_probe = {
            **before_probe,
            "ok": True,
            "failure_kind": None,
            "available_memory_mb": 32768.0,
            "matched_granularity_mb": 2,
        }
        blocked = MODULE.evaluate_memory_scan(self.rule("MEM-001"), {"memory_scan": before_probe})
        context = MODULE.CheckContext("A3", "memcache", 32768)
        with (
            patch.object(MODULE.os, "geteuid", return_value=0),
            patch.object(MODULE, "run_cmd", return_value=(0, "", "")) as sync_mock,
            patch.object(MODULE.Path, "write_text", return_value=2) as write_mock,
            patch.object(MODULE, "probe_memory_capacity", return_value=after_probe) as scan_mock,
        ):
            final_results, actions = MODULE.execute_fix(self.catalog, context, [blocked])
        self.assertEqual(final_results[0]["status"], "PASS")
        self.assertEqual(actions[0]["status"], "SUCCESS")
        self.assertEqual(actions[0]["before_evidence"]["available_memory_mb"], 0.0)
        self.assertEqual(actions[0]["after_evidence"]["matched_granularity_mb"], 2)
        sync_mock.assert_called_once_with(["sync"], timeout=60)
        write_mock.assert_called_once_with("3\n", encoding="ascii")
        scan_mock.assert_called_once_with("A3", 32768)

    def test_memory_fix_does_not_mutate_without_root(self):
        probe = {
            "method": "a3_mem_scan",
            "ok": False,
            "script": "mem_scan.py",
            "scan_scope": "a3_fixed_ranges",
            "managed_memory_mb": 32768,
            "failure_kind": "capacity",
            "available_memory_mb": 0.0,
            "matched_granularity_mb": None,
            "scans": {},
            "error": None,
        }
        blocked = MODULE.evaluate_memory_scan(self.rule("MEM-001"), {"memory_scan": probe})
        context = MODULE.CheckContext("A3", "memcache", 32768)
        with (
            patch.object(MODULE.os, "geteuid", return_value=1000),
            patch.object(MODULE, "run_cmd") as sync_mock,
            patch.object(MODULE.Path, "write_text") as write_mock,
            patch.object(MODULE, "probe_memory_capacity") as scan_mock,
        ):
            final_results, actions = MODULE.execute_fix(self.catalog, context, [blocked])
        self.assertEqual(final_results, [blocked])
        self.assertEqual(actions[0]["status"], "FAILED")
        sync_mock.assert_not_called()
        write_mock.assert_not_called()
        scan_mock.assert_not_called()

    def test_text_exposes_fix_and_repair_result(self):
        blocked = MODULE.evaluate_manual_confirmation(self.rule("LINGQU-001"), {"manual_confirmation": {}})
        action = {
            "id": "LINGQU-001",
            "type": "manual_switch_upgrade",
            "status": "MANUAL",
            "message": blocked["remediation"]["description"],
        }
        report = MODULE.render_text(self.payload([blocked], fix=True, actions=[action]))
        self.assertIn("- fix: true", report)
        self.assertIn("Repair results: SUCCESS=0, NO_EFFECT=0, FAILED=0, SKIPPED=0, MANUAL=1", report)
        self.assertIn(f"- LINGQU-001 MANUAL: {action['message']}", report)


if __name__ == "__main__":
    unittest.main()
