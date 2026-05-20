# settings

PY := python3.8
ENV_NAME := .venv
DELETION_DELAY ?= 10

# computed

ACTIV_ENV := . $(ENV_NAME)/bin/activate
IN_ENV := $(ACTIV_ENV) &&

# includes
-include *.mk

# actions

help:
	$(PY) main.py -h

test:
	@echo "\n[[ running correctness tests ]]"
	$(IN_ENV) python -m pytest test/ -v

activate:
	# warning: will not work (limitation of `make`), you need to run it manually
	$(ACTIV_ENV)

py_init:
	@echo "\n[[ creating python environment if absent ]]"
	$(PY) -m venv $(ENV_NAME)

py_dep: py_init
	@echo "\n[[ installing/upgrading python dependencies ]]"
	$(IN_ENV) python3 -m pip install --upgrade pip
	$(IN_ENV) pip install --upgrade $(shell grep -v "^z3log\|^#\|^$$" requirements.txt | tr '\n' ' ')
	$(IN_ENV) pip install --no-deps z3log==2.2.13
	$(MAKE) pygraphviz_dep

pygraphviz_dep: py_init
	@echo "\n[[ installing pygraphviz with graphviz 14.x patch ]]"
	$(eval PGVTMP := $(shell mktemp -d))
	$(IN_ENV) pip download pygraphviz==1.10 --no-deps -d $(PGVTMP)
	cd $(PGVTMP) && unzip -q pygraphviz-1.10.zip
	sed -i 's/unsigned int \*arg5 = (unsigned int \*) 0 ;/size_t *arg5 = (size_t *) 0 ;/' $(PGVTMP)/pygraphviz-1.10/pygraphviz/graphviz_wrap.c
	sed -i 's/unsigned int tempn4 ;/size_t tempn4 ;/' $(PGVTMP)/pygraphviz-1.10/pygraphviz/graphviz_wrap.c
	$(IN_ENV) pip install $(PGVTMP)/pygraphviz-1.10
	rm -rf $(PGVTMP)

setup: py_init py_dep

rm_data:
	@echo "\n[[   deleting all final and intermediary data   ]]\n[[ YOU HAVE $(DELETION_DELAY) SECONDS TO CANCEL THIS OPERATION ]]"
	@sleep $(DELETION_DELAY)
	rm -rf output/ test/

rm_cache:
	@echo "\n[[ removing all pycache folders ]]"
	find . -name __pycache__ -prune -print -exec rm -rf {} \;

rm_temp:
	@echo "\n[[ removing generated temporary files ]]"
	rm -f yosys_graph.log .history_sta

rm_pyenv:
	@echo "\n[[ removing the virtual python environment ]]"
	rm -rf $(ENV_NAME)

clean: rm_cache rm_temp

clean-all: rm_pyenv rm_data clean
