#!/bin/bash
set -x

SDIR=$(dirname "$0")

. "$SDIR/django_db_settings.sh"

db_settings | (
    IFS='|'
    read DBTYPE HOST USER PASS DBNAME

    if [ "$DBTYPE" != "mysql" ]; then
        echo "Django is setup to use $DBTYPE instead of mysql database." >&2
        exit 2
    fi

    mysqldump --skip-extended-insert --compact -c \
        -u "$USER" -p"$PASS" -h "$HOST" "$DBNAME"
)
