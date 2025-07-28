#!/bin/bash
export BINDIR="/cygdrive/c/Users/500138/Documents/databases/nspe57/"
for f in *.in;do
	echo $f
	cp $f nspe.in
	$BINDIR/nspe.x nspe.in
	fbase=${f%.in}
	cp nspe01.asc "${fbase}_01.asc"
	cp nspe03.asc "${fbase}_03.asc"
	cp for003.dat "${fbase}.ff"
	rm pade_coeff.txt for*.dat for*.bin *.log
done;
