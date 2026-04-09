PYTHON     := python3
DEMO13_DIR := 13-OAT-SENSITIVITY
DEMO14_DIR := 14-OAT-SENSITIVITY-DRAINED-SIDE

.PHONY: run_13_14

run_13_14:
	@echo "=== [1/5] Running Demo 13 (sealed side) ==="
	cd $(DEMO13_DIR) && $(PYTHON) run_all.py
	@echo "=== [2/5] OAT sensitivity plot — Demo 13 ==="
	cd $(DEMO13_DIR) && $(PYTHON) plot_comparison.py
	@echo "=== [3/5] Running Demo 14 (drained side) ==="
	cd $(DEMO14_DIR) && $(PYTHON) run_all.py
	@echo "=== [4/5] OAT sensitivity plot — Demo 14 ==="
	cd $(DEMO14_DIR) && $(PYTHON) plot_comparison.py
	@echo "=== [5/6] Comparing Demo 13 vs Demo 14 ==="
	$(PYTHON) compare_13_14.py
	@echo "=== [6/7] P-Q plots for Demo 14 ==="
	cd $(DEMO14_DIR) && $(PYTHON) plot_pq.py
	@echo "=== [7/8] UCS failure table for Demo 14 ==="
	cd $(DEMO14_DIR) && $(PYTHON) table_ucs_failure.py
	@echo "=== [8/8] UCS / p''_t sweep tile plot for Demo 14 ==="
	cd $(DEMO14_DIR) && $(PYTHON) plot_ucs_table.py
	@echo "=== Done ==="
