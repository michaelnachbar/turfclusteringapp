#!/usr/bin/env python
import os
import sys
import sqlite3

"""
Given an sqlite database find all json columns and try to compress them if large enough.
"""

import json
try:
    import lzma
except:
    from backports import lzma
LZMA_HDR = '\xca\xfeLZ'

def adapt_json(data):
    data = (json.dumps(data, sort_keys=True)).encode()
    return buffer(LZMA_HDR+lzma.compress(data, format=lzma.FORMAT_ALONE, preset=9))

def convert_json(blob):
    if blob[:4] == LZMA_HDR:
        blob = lzma.decompress(blob[4:])
    return json.loads(blob.decode())

sqlite3.register_adapter(dict, adapt_json)
sqlite3.register_adapter(list, adapt_json)
sqlite3.register_adapter(tuple, adapt_json)
sqlite3.register_converter(str('JSON'), convert_json)


def gen_tables_names(cur):
    res = cur.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()
    return (r[0] for r in res)

def gen_json_fields(cur):
    for table in gen_tables_names(cur):
        res = cur.execute('PRAGMA table_info("{}")'.format(table))
        for col_spec in res:
            if col_spec[2].lower() == 'json':
                yield (table, col_spec[1])

def main(argv):
    sqlitedb =  argv[0]
    
    # Set the sqlite temp directory to the current one, if not set
    if 'SQLITE_TMPDIR' not in os.environ:
        os.putenv('SQLITE_TMPDIR', os.getcwd());
    
    conn = sqlite3.connect(sqlitedb, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    cur = conn.cursor()
    
    for table, colname in gen_json_fields(cur):
        maxrows = cur.execute('SELECT MAX(oid) FROM "{}";'.format(table)).fetchone()[0]
        chunksz = 10000
        min_len = 100
        
        print 'Compressing json column {}.{} with {} rows'.format(table, colname, maxrows)
        
        format_dict = dict(table=table, colname=colname, lzma_hdr=LZMA_HDR)
        
        for i in xrange(0, maxrows, chunksz):
            # For each chunk of row ids, get all rows that are above min_len
            # and have not already been compressed.
            res = cur.execute('''
                SELECT oid, "{colname}" FROM "{table}"
                    WHERE oid >= ? AND oid < ? AND
                          length("{colname}") > ? AND
                          typeof("{colname}") != 'blob' AND
                          CAST("{colname}" AS text) NOT LIKE ?||"%";
                              '''.format(**format_dict),
                              (i, i+chunksz, min_len, sqlite3.Binary(LZMA_HDR))).fetchall()
            
            for oid, json_data in res:
                cur.execute('UPDATE "{table}" SET "{colname}" = ? WHERE oid = ?;'.format(**format_dict),
                            (sqlite3.Binary(adapt_json(json_data)), oid))
            
            conn.commit()
    
    cur.execute('VACUUM;')


if __name__ == "__main__":
    main(sys.argv[1:])
