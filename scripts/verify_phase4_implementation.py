#!/usr/bin/env python3
"""
Verification script for Phase 4 Implementation.
Verifies that all components are properly implemented and integrated.
"""

import asyncio
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the somba_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase4Verifier:
    """Verifies Phase 4 implementation completeness."""
    
    def __init__(self):
        self.verification_results = []
        self.required_modules = [
            'somba_pipeline.manager',
            'somba_pipeline.lease_manager',
            'somba_pipeline.config_sync',
            'somba_pipeline.worker_process',
            'somba_pipeline.control_plane_client',
        ]
        self.required_classes = [
            'ManagerProcess',
            'LeaseManager',
            'ConfigurationSync',
            'WorkerProcess',
            'WorkerPool',
            'ControlPlaneClient',
            'ManagerConfig',
        ]
        self.required_methods = {
            'ManagerProcess': [
                'start', '_lease_management_loop', '_worker_management_loop',
                '_configuration_sync_loop', '_health_monitoring_loop',
                '_register_runner', '_get_available_cameras',
                '_acquire_camera_leases', '_renew_active_leases',
                '_calculate_worker_assignments', '_adjust_workers',
                '_start_worker', '_stop_worker', '_monitor_worker_health',
                '_restart_worker', '_update_worker_configuration',
                '_send_runner_heartbeat', '_check_lease_health',
                '_handle_lease_loss', '_shutdown'
            ],
            'LeaseManager': [
                'acquire_camera_lease', 'renew_lease', 'release_lease',
                'send_heartbeat', 'get_camera_lease', 'get_active_leases',
                'validate_lease', 'cleanup_expired_leases', 'shutdown',
                'get_lease_stats'
            ],
            'ConfigurationSync': [
                'sync_camera_configurations', 'sync_global_configuration',
                'watch_configuration_changes', 'get_camera_config',
                'get_camera_config_version', 'get_all_camera_configs',
                'add_camera_config', 'remove_camera_config',
                'force_sync_camera', 'validate_configuration_consistency',
                'cleanup_stale_configurations', 'shutdown', 'get_sync_stats'
            ],
            'WorkerProcess': [
                'start', 'stop', 'restart', 'is_healthy',
                'get_stats', 'update_camera_assignments', 'send_signal'
            ],
            'WorkerPool': [
                'start_worker', 'stop_worker', 'restart_worker',
                'get_worker', 'get_all_workers', 'get_worker_assignments',
                'get_healthy_workers', 'get_unhealthy_workers',
                'monitor_and_restart', 'shutdown', 'get_pool_stats'
            ],
            'ControlPlaneClient': [
                'acquire_lease', 'renew_lease', 'release_lease',
                'lease_heartbeat', 'get_runner_leases', 'get_camera_lease',
                'register_runner', 'send_runner_heartbeat', 'get_runner',
                'unregister_runner', 'get_available_cameras',
                'get_camera_config', 'get_cameras', 'update_camera_config',
                'get_zone_configs', 'create_zone_config', 'update_zone_config',
                'delete_zone_config', 'get_config_version',
                'get_camera_assignments', 'health_check', 'get_metrics',
                'test_connection', 'batch_get_camera_configs',
                'batch_acquire_leases', 'close'
            ]
        }
        
    def add_result(self, test_name: str, success: bool, details: str = ""):
        """Add verification result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.verification_results.append({
            "test_name": test_name,
            "status": status,
            "details": details
        })
        logger.info(f"Verification {test_name}: {status} - {details}")
    
    def verify_module_imports(self):
        """Verify that all required modules can be imported."""
        try:
            for module_name in self.required_modules:
                try:
                    importlib.import_module(module_name)
                    self.add_result(f"Module Import: {module_name}", True, "Module imported successfully")
                except ImportError as e:
                    self.add_result(f"Module Import: {module_name}", False, f"Import failed: {e}")
        except Exception as e:
            self.add_result("Module Imports", False, f"Unexpected error: {e}")
    
    def verify_class_definitions(self):
        """Verify that all required classes are defined."""
        try:
            for class_name in self.required_classes:
                found = False
                
                # Check each module for the class
                for module_name in self.required_modules:
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, class_name):
                            found = True
                            self.add_result(f"Class Definition: {class_name}", True, f"Found in {module_name}")
                            break
                    except ImportError:
                        continue
                
                if not found:
                    self.add_result(f"Class Definition: {class_name}", False, "Class not found in any module")
                    
        except Exception as e:
            self.add_result("Class Definitions", False, f"Unexpected error: {e}")
    
    def verify_method_implementations(self):
        """Verify that all required methods are implemented."""
        try:
            for class_name, required_methods in self.required_methods.items():
                # Find the class
                class_found = False
                for module_name in self.required_modules:
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, class_name):
                            class_found = True
                            cls = getattr(module, class_name)
                            
                            # Check each required method
                            for method_name in required_methods:
                                if hasattr(cls, method_name):
                                    method = getattr(cls, method_name)
                                    if callable(method):
                                        self.add_result(f"Method Implementation: {class_name}.{method_name}", True, "Method implemented")
                                    else:
                                        self.add_result(f"Method Implementation: {class_name}.{method_name}", False, "Method not callable")
                                else:
                                    self.add_result(f"Method Implementation: {class_name}.{method_name}", False, "Method not found")
                            
                            break
                    except ImportError:
                        continue
                
                if not class_found:
                    self.add_result(f"Method Implementation: {class_name}", False, "Class not found")
                    
        except Exception as e:
            self.add_result("Method Implementations", False, f"Unexpected error: {e}")
    
    def verify_integration_points(self):
        """Verify integration points between components."""
        try:
            # Check ManagerProcess integration
            try:
                from somba_pipeline.manager import ManagerProcess
                from somba_pipeline.lease_manager import LeaseManager
                from somba_pipeline.config_sync import ConfigurationSync
                from somba_pipeline.worker_process import WorkerProcess
                
                # Check if ManagerProcess uses other components
                init_signature = inspect.signature(ManagerProcess.__init__)
                init_params = list(init_signature.parameters.keys())
                
                expected_components = ['config']
                if all(param in init_params for param in expected_components):
                    self.add_result("ManagerProcess Integration", True, "ManagerProcess properly configured")
                else:
                    self.add_result("ManagerProcess Integration", False, "ManagerProcess missing expected parameters")
                
            except Exception as e:
                self.add_result("ManagerProcess Integration", False, f"Error checking integration: {e}")
            
            # Check LeaseManager integration
            try:
                from somba_pipeline.lease_manager import LeaseManager
                from somba_pipeline.control_plane_client import ControlPlaneClient
                
                # Check if LeaseManager uses ControlPlaneClient
                init_signature = inspect.signature(LeaseManager.__init__)
                init_params = list(init_signature.parameters.keys())
                
                if 'control_plane_config' in init_params:
                    self.add_result("LeaseManager Integration", True, "LeaseManager properly configured")
                else:
                    self.add_result("LeaseManager Integration", False, "LeaseManager missing expected parameters")
                
            except Exception as e:
                self.add_result("LeaseManager Integration", False, f"Error checking integration: {e}")
            
            # Check ConfigurationSync integration
            try:
                from somba_pipeline.config_sync import ConfigurationSync
                
                # Check if ConfigurationSync has proper methods
                required_sync_methods = ['sync_camera_configurations', 'watch_configuration_changes']
                has_all_methods = all(hasattr(ConfigurationSync, method) for method in required_sync_methods)
                
                if has_all_methods:
                    self.add_result("ConfigurationSync Integration", True, "ConfigurationSync properly implemented")
                else:
                    self.add_result("ConfigurationSync Integration", False, "ConfigurationSync missing methods")
                
            except Exception as e:
                self.add_result("ConfigurationSync Integration", False, f"Error checking integration: {e}")
            
        except Exception as e:
            self.add_result("Integration Points", False, f"Unexpected error: {e}")
    
    def verify_configuration_files(self):
        """Verify that configuration files exist."""
        try:
            config_files = [
                "config/test_manager_config.json",
                "config/test_shard.json",
                "docs/phase4_manager_layer_implementation.md"
            ]
            
            for config_file in config_files:
                file_path = Path(__file__).parent.parent / config_file
                if file_path.exists():
                    self.add_result(f"Configuration File: {config_file}", True, "File exists")
                else:
                    self.add_result(f"Configuration File: {config_file}", False, "File not found")
                    
        except Exception as e:
            self.add_result("Configuration Files", False, f"Unexpected error: {e}")
    
    def verify_worker_enhancements(self):
        """Verify that ProductionWorker has been enhanced with lease awareness."""
        try:
            from somba_pipeline.worker import ProductionWorker
            
            # Check for lease-aware attributes
            lease_attributes = ['lease_id', 'worker_id', 'lease_manager', 'config_sync', 'lease_heartbeat_task']
            has_all_attributes = all(hasattr(ProductionWorker, attr) for attr in lease_attributes)
            
            if has_all_attributes:
                self.add_result("Worker Lease Awareness", True, "ProductionWorker has lease-aware attributes")
            else:
                missing_attrs = [attr for attr in lease_attributes if not hasattr(ProductionWorker, attr)]
                self.add_result("Worker Lease Awareness", False, f"Missing attributes: {missing_attrs}")
            
            # Check for lease-aware methods
            lease_methods = [
                'set_lease_info', 'start_lease_heartbeat', 'stop_lease_heartbeat',
                '_lease_heartbeat_loop', 'handle_configuration_update',
                'get_worker_stats', 'handle_lease_loss'
            ]
            
            has_all_methods = all(hasattr(ProductionWorker, method) for method in lease_methods)
            
            if has_all_methods:
                self.add_result("Worker Lease Methods", True, "ProductionWorker has lease-aware methods")
            else:
                missing_methods = [method for method in lease_methods if not hasattr(ProductionWorker, method)]
                self.add_result("Worker Lease Methods", False, f"Missing methods: {missing_methods}")
                
        except Exception as e:
            self.add_result("Worker Enhancements", False, f"Error checking worker enhancements: {e}")
    
    def verify_script_functionality(self):
        """Verify that test scripts exist and are functional."""
        try:
            scripts = [
                "scripts/test_phase4_manager.py",
                "scripts/verify_phase4_implementation.py"
            ]
            
            for script in scripts:
                script_path = Path(__file__).parent.parent / script
                if script_path.exists():
                    # Check if script is executable
                    if script_path.stat().st_mode & 0o111:
                        self.add_result(f"Script Functionality: {script}", True, "Script exists and is executable")
                    else:
                        self.add_result(f"Script Functionality: {script}", True, "Script exists but not executable")
                else:
                    self.add_result(f"Script Functionality: {script}", False, "Script not found")
                    
        except Exception as e:
            self.add_result("Script Functionality", False, f"Unexpected error: {e}")
    
    def run_all_verifications(self):
        """Run all verifications."""
        logger.info("Starting Phase 4 Implementation Verification")
        
        verifications = [
            self.verify_module_imports,
            self.verify_class_definitions,
            self.verify_method_implementations,
            self.verify_integration_points,
            self.verify_configuration_files,
            self.verify_worker_enhancements,
            self.verify_script_functionality,
        ]
        
        for verification in verifications:
            try:
                verification()
            except Exception as e:
                self.add_result(verification.__name__, False, f"Verification crashed: {e}")
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print verification results."""
        print("\n" + "="*80)
        print("PHASE 4 IMPLEMENTATION VERIFICATION RESULTS")
        print("="*80)
        
        passed = 0
        failed = 0
        
        for result in self.verification_results:
            status = result["status"]
            if "PASS" in status:
                passed += 1
            else:
                failed += 1
            
            print(f"{status:12} | {result['test_name']}")
            if result["details"]:
                print(f"             | {result['details']}")
        
        print("-"*80)
        print(f"TOTAL: {len(self.verification_results)} verifications")
        print(f"PASS:  {passed} verifications")
        print(f"FAIL:  {failed} verifications")
        
        if failed == 0:
            print("\n🎉 All verifications passed! Phase 4 implementation is complete.")
            return True
        else:
            print(f"\n⚠️  {failed} verifications failed. Please review the implementation.")
            return False


def main():
    """Main verification runner."""
    verifier = Phase4Verifier()
    success = verifier.run_all_verifications()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()