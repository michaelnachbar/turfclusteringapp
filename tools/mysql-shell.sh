#!/bin/bash
set -x

SDIR=$(dirname "$0")

. "$SDIR/django_db_settings.sh"

db_settings | (
    IFS='|'
    read DBTYPE HOST USER PASS DBNAME
    
    mysql -c -u "$USER" -p"$PASS" -h "$HOST" "$DBNAME" </dev/tty
)