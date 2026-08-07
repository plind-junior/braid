.PHONY: provision test-remote bench-noise bench-scaling gpu-start gpu-stop gpu-status

# The box bills ~$0.79/hr while running and storage-only while stopped, so stop
# it as soon as a batch of GPU work is done. Always `stop`, never `destroy`:
# stop preserves the 300 GB disk holding ~13 GB of staged models and the built
# llama.cpp tree. Destroying costs a re-download and a ~10 min rebuild.
VAST_INSTANCE ?= 47055458

gpu-start:
	vastai start instance $(VAST_INSTANCE)

gpu-stop:
	vastai stop instance $(VAST_INSTANCE)

# `uptime` inside the container reports the HOST kernel's uptime, not the
# rental duration. `duration` from the API is the real number.
gpu-status:
	@vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; \
	d=json.load(sys.stdin); d=d if isinstance(d,list) else d.get('instances',[]); \
	[print('id=%s label=%s status=%s \$$/hr=%s rented_hrs=%.1f' % (i.get('id'), \
	i.get('label'), i.get('actual_status'), i.get('dph_total'), \
	(i.get('duration') or 0)/3600)) for i in d]"

provision:
	./scripts/remote.sh bash scripts/provision_remote.sh

# ARGS replaces the default target rather than appending to it, so
# `make test-remote ARGS="tests/foo.py -s"` runs exactly that file.
test-remote:
	./scripts/remote.sh python3 -B -m pytest $(if $(ARGS),$(ARGS),tests/)

bench-noise:
	./scripts/remote.sh python3 -B -m braid.bench.noise_floor

bench-scaling:
	./scripts/remote.sh python3 -B -m braid.bench.scan_scaling
