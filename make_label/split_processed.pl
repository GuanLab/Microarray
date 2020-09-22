#!/usr/bin/perl
#
open OLD, "all_annotation_aug30.txt_processed" or die;
$line=<OLD>;
chomp $line;
@table=split "\t", $line;
$ind=$table[0];
system "mkdir split_processed";
open NEW, ">split_processed/$ind" or die;
print NEW "$line\n";

while ($line=<OLD>){
	chomp $line;
	@table=split "\t", $line;
	if ($table[0] eq $ind){
		print NEW "$line\n";
	}else{
		close NEW;
		$ind=$table[0];
		open NEW, ">split_processed/$ind" or die;
		print NEW "$line\n";
	}
}
close NEW;
close OLD;

