# -*- coding: utf-8 -*-

import os
import re

root = 'C:/tmp/20news-bydate/'
# files = root + '20news-bydate-train/'
files = root + '20news-bydate-test/'

# output = file('C:/tmp/20news-bydate/train.tab', 'w')
output = file('C:/tmp/20news-bydate/test.tab', 'w')

for cls in os.listdir(files):
    print 'processing', cls, '...'
    cur_dir = files + cls + '/'
    all_files = os.listdir(cur_dir)
    
    for f in all_files:
        content = file(cur_dir + f).read()
        cut = content.index('\n\n')

        header_str = content[:cut].split('\n')
        header = dict(s.split(': ', 1) for s in header_str if ': ' in s)
        title = re.sub('\s+', ' ', header['Subject'])
        
        body = content[cut+2:]
        body = re.sub('\s+', ' ', body)
        
        output.write(cls + '\t' + title + '\t' + body)
        output.write('\n')
        
output.close()
