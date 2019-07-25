#!/bin/bash

SDIR=$(dirname "$0")

if [ ! -f "$SDIR/../canvas_cutting/local_settings.py" ]; then
    echo "The DATABASES in local settings must be setup first." >&2
    exit 1
fi

function db_settings() {(
    cd "$SDIR/../canvas_cutting"
    python <(cat <<'EOF'
from __future__ import print_function
import sys, os
sys.path.insert(0, os.getcwd())
import local_settings as l
db=l.DATABASES.get("default", {})
print('|'.join((db.get("ENGINE", '').split(".")[-1],db.get("HOST",''),db.get("USER",''),
      db.get("PASSWORD",''),db.get("NAME",''))))
EOF
    )
) || exit $?
}