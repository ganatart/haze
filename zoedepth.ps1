git clone https://github.com/isl-org/ZoeDepth
mv .\ZoeDepth\zoedepth\* .\tmp
rm -r -Force .\ZoeDepth
mkdir zoedepth
mv .\tmp\* .\zoedepth
rm -r -Force .\tmp