#!/usr/bin/perl

open OLD, "all_annotation_aug30.txt" or die;
open NEW, ">all_annotation_aug30.txt_processed" or die;
while ($line=<OLD>){
    chomp $line;
    @table=split '\t', $line;
    @t=split '/', $table[0];
    print NEW "$t[-1]";


    @table=split '"x": ', $line;
    @table=split '}', $table[1];
    @table=split ']', $table[0];
    @table=split ',', $table[0];
    print NEW "\t$table[0]";

    @table=split '"y": ', $line;
    @table=split '}', $table[1];
    @table=split ']', $table[0];
    @table=split ',', $table[0];
    print NEW "\t$table[0]";

    @table=split '"width": ', $line;
    @table=split '}', $table[1];
    @table=split ']', $table[0];
    @table=split ',', $table[0];
    print NEW "\t$table[0]";
    

    @table=split '"height": ', $line;
    @table=split '}', $table[1];
    @table=split ']', $table[0];
    @table=split ',', $table[0];
    print NEW "\t$table[0]\n";
}

