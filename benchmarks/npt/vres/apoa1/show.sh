#!/bin/bash -e

grep dynamics p008.node001* | awk '{print $4}'
grep dynamics p008.node002* | awk '{print $4}'
grep dynamics p008.node004* | awk '{print $4}'
grep dynamics p008.node008* | awk '{print $4}'

grep dynamics p016.node002* | awk '{print $4}'
grep dynamics p016.node004* | awk '{print $4}'
grep dynamics p016.node008* | awk '{print $4}'
grep dynamics p016.node016* | awk '{print $4}'

grep dynamics p032.node004* | awk '{print $4}'
grep dynamics p032.node008* | awk '{print $4}'
grep dynamics p032.node016* | awk '{print $4}'
grep dynamics p032.node032* | awk '{print $4}'

grep dynamics p064.node008* | awk '{print $4}'
grep dynamics p064.node016* | awk '{print $4}'
grep dynamics p064.node032* | awk '{print $4}'
grep dynamics p064.node064* | awk '{print $4}'
