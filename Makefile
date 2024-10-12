RANGE = 1 2 3 4 5 6 7 8 9 10 11 12 13 14

.PHONY: all

# デフォルトのターゲット
all:
	@echo "使用方法: make run YYYYMMDD"

# メインの実行ターゲット
run:
ifeq ($(words $(MAKECMDGOALS)),1)
	@echo "エラー: 日付が指定されていません。使用方法: make run YYYYMMDD"
else
	$(eval DATE := $(word 2,$(MAKECMDGOALS)))
	@for i in $(RANGE); do \
		echo "実行中: python pr8.py $(DATE) $$i"; \
		python pr8.py $(DATE) $$i; \
	done
endif

# 個別のターゲット
run_%:
ifeq ($(words $(MAKECMDGOALS)),1)
	@echo "エラー: 日付が指定されていません。使用方法: make run_$* YYYYMMDD"
else
	$(eval DATE := $(word 2,$(MAKECMDGOALS)))
	@echo "実行中: python pr8.py $(DATE) $*"
	@python pr8.py $(DATE) $*
endif

%:
	@: