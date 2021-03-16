(* ::Package:: *)

BeginPackage["PMTools`"];

EndPackage[];



Begin["PMTools`Private`"];

(** PackageInfo **)

$PackageUsageInfo = {"object" -> "This is the head used for generic objects.", "profile" -> "This is the head used for profile objects.", "CallGraph" -> "???", "CallsTo" -> "???", "CallsFrom" -> "???", "CallsWith" -> "???", "CallsTo" -> "???", "CallsWith" -> "???", "CallsFrom" -> "???", "FindNotebooks" -> "???", "DeclutterNotebooks" -> "???", "DeclutterNotebooks" -> "???", "PackageSourceDeclutterNotebooks" -> "???", "fixBSplineFunction" -> "???", "MemoryStatistics" -> "MemoryStatistics[M] returns the memory consumption of object M.", "MemoryChart" -> "MemoryChart[M] plots a pie chart that visualizes the memory consumption of object M.", "OpenSource" -> "???", "OpenSource" -> "???", "OpenSource" -> "???", "OpenSource" -> "???", "PackageSourceBackup" -> "???", "PackageSourceCopyRequired" -> "???", "PackageSourceFindCommand" -> "???", "PackageSourceFindCommand" -> "???", "PackageSourceReplaceCommand" -> "???", "NumberOfCalls" -> "???", "Data" -> "???", "TotalTime" -> "???", "Eigentimes" -> "???", "EigentimeTable" -> "???", "EigentimeChart" -> "???", "EigentimeChart" -> "???", "SunburstLeaves" -> "???", "SunburstData" -> "???", "Sunburst" -> "???", "Sunburst" -> "???", "$EigenIncludeDirectory" -> "", "$OpenMPIncludeDirectory" -> "", "$CountingData" -> "", "CountingReap" -> "", "CreateFormat" -> "", "Init" -> "Init[object] returns an empty object.", "Inherit" -> "Init[object] returns an empty object.", "Init" -> "", "DebuggingReap" -> "", "TimingReap" -> ""};

CreateUsageMessage[name_String, message_String] := Module[{context, cname, str}, Quiet[context = Context[Evaluate[name]]]; If[Head[context] === String, cname = StringJoin[context, name], cname = name]; With[{musage = ToExpression[StringJoin["MessageName[", cname, ",\"usage\"]"]]}, If[Head[musage] === MessageName, str = StringJoin[cname, "::usage = ", ToString[message, InputForm], ";"]; ToExpression[str]; , If[StringPosition[musage, message] == {}, str = StringJoin[cname, "::usage = ", ToString[StringJoin[musage, "\n\n", message], InputForm], ";"]; ToExpression[str]]]]; ];

End[];



Begin["PMTools`"];

(** Public Declarations **)

PMTools`Private`CreateUsageMessage@@@PMTools`Private`$PackageUsageInfo

$packageDirectory  = DirectoryName[$InputFileName];



End[];

Begin["PMTools`Private`"];

TimingSow = #1&;

(** Objects **)

object /: ClearAllCache[PM`Private`x_object, PM`Private`arg_:All] := (ClearCache[PM`Private`x, PM`Private`arg]; ClearPersistentCache[PM`Private`x, PM`Private`arg]; );

object /: ClearAllCacheDependingOn[PM`Private`x_object, PM`Private`s_] := ClearAllCache[PM`Private`x, VertexInComponent[CallGraph[$PM], ToString /@ Flatten[{PM`Private`s}]]];

object /: Cache[PM`Private`x:object[PM`Objects`$object_], args___] := PM`Objects`$object[["Cache",args]];

object /: CacheKeys[PM`Private`x:object[PM`Objects`$object_]] := Keys[PM`Objects`$object[["Cache"]]];

object /: ClearCache[PM`Private`x:object[PM`Objects`$object_]] := (PM`Objects`$object[["Cache"]] = Association[]; );

object /: ClearCache[PM`Private`x:object[PM`Objects`$object_], All] := (PM`Objects`$object[["Cache"]] = Association[]; );

object /: ClearCache[PM`Private`x:object[PM`Objects`$object_], PM`Private`s_String] := KeyDropFrom[PM`Objects`$object[["Cache"]], PM`Private`s];

object /: ClearCache[PM`Private`x:object[PM`Objects`$object_], PM`Private`p_] := KeyDropFrom[PM`Objects`$object[["Cache"]], ToString[PM`Private`p, InputForm]];

object /: ClearCache[PM`Private`x:object[PM`Objects`$object_], {PM`Private`s___String}] := KeyDropFrom[PM`Objects`$object[["Cache"]], {PM`Private`s}];

object /: ClearCache[PM`Private`x:object[PM`Objects`$object_], {PM`Private`p__}] := KeyDropFrom[PM`Objects`$object[["Cache"]], Function[PM`Private`z, If[StringQ[PM`Private`z], PM`Private`z, ToString[PM`Private`z, InputForm]]] /@ Flatten[{PM`Private`p}]];

object /: ClearCacheDependingOn[PM`Private`x_object, PM`Private`s_] := ClearCache[PM`Private`x, VertexInComponent[CallGraph[$PM], ToString /@ Flatten[{PM`Private`s}]]];

object /: SetCache[PM`Private`x:object[PM`Objects`$object_], PM`Private`pos_, PM`Private`val_] := (PM`Objects`$object[["Cache",PM`Private`pos]] = PM`Private`val; );

object /: PersistentCache[PM`Private`x:object[PM`Objects`$object_], args___] := PM`Objects`$object[["PersistentCache",args]];

object /: PersistentCacheKeys[PM`Private`x:object[PM`Objects`$object_]] := Keys[PM`Objects`$object[["PersistentCache"]]];

object /: ClearPersistentCache[PM`Private`x:object[PM`Objects`$object_]] := (PM`Objects`$object[["PersistentCache"]] = Association[]; );

object /: ClearPersistentCache[PM`Private`x:object[PM`Objects`$object_], All] := (PM`Objects`$object[["PersistentCache"]] = Association[]; );

object /: ClearPersistentCache[PM`Private`x:object[PM`Objects`$object_], PM`Private`s_String] := KeyDropFrom[PM`Objects`$object[["PersistentCache"]], PM`Private`s];

object /: ClearPersistentCache[PM`Private`x:object[PM`Objects`$object_], PM`Private`p_] := KeyDropFrom[PM`Objects`$object[["PersistentCache"]], ToString[PM`Private`p, InputForm]];

object /: ClearPersistentCache[PM`Private`x:object[PM`Objects`$object_], {PM`Private`s___String}] := KeyDropFrom[PM`Objects`$object[["PersistentCache"]], {PM`Private`s}];

object /: ClearPersistentCache[PM`Private`x:object[PM`Objects`$object_], {PM`Private`p__}] := KeyDropFrom[PM`Objects`$object[["PersistentCache"]], Function[PM`Private`z, If[StringQ[PM`Private`z], PM`Private`z, ToString[PM`Private`z, InputForm]]] /@ Flatten[{PM`Private`p}]];

object /: ClearPersistentCacheDependingOn[PM`Private`x_object, PM`Private`s_] := ClearPersistentCache[PM`Private`x, VertexInComponent[CallGraph[$PM], ToString /@ Flatten[{PM`Private`s}]]];

object /: SetPersistentCache[PM`Private`x:object[PM`Objects`$object_], PM`Private`pos_, PM`Private`val_] := (PM`Objects`$object[["PersistentCache",PM`Private`pos]] = PM`Private`val; );

SetAttributes[object, HoldAll];

ToExpression[StringJoin["PM`Private`$", "object", "Counter"], InputForm, Function[PM`Private`data, PM`Private`data = 0, HoldAll]];

Options[object] = {"CacheActivated" -> True, "PersistentCacheActivated" -> True};

object /: DeepCopy[PM`Private`x:object[PM`Objects`$object_]] := Initialize[object, DeepCopy /@ PM`Objects`$object];

object /: (PM`Private`y_) \[LeftArrow] (PM`Private`x:object[PM`Objects`$nobject_]) := PM`Private`y = DeepCopy[PM`Private`x];

object /: (PM`Private`x:object[PM`Objects`$object_])[[1,args__]] := PM`Objects`$object[[args]];

object /: (PM`Private`x:object[PM`Objects`$object_])[[PM`Private`s_String,args___]] := PM`Objects`$object[[PM`Private`s,args]];

object /: (PM`Private`x_object)[PM`Private`s_String, args___] := PM`Private`x[[1]][[PM`Private`s,args]];

object /: Initialize[object, PM`Private`data0_Association] := With[{PM`Private`c = ToExpression[StringJoin["++PM`Private`$", "object", "Counter"]]}, ToExpression[StringJoin["PM`Objects`$", "object", "$", ToString[PM`Private`c]], InputForm, Function[PM`Private`data, SetAttributes[PM`Private`data, Temporary]; PM`Private`data = PM`Private`data0; If[ !KeyExistsQ[PM`Private`data, "Cache"], AppendTo[PM`Private`data, "Cache" -> Association[]]]; If[ !KeyExistsQ[PM`Private`data, "PersistentCache"], AppendTo[PM`Private`data, "PersistentCache" -> Association[]]]; If[ !KeyExistsQ[PM`Private`data, "Settings"], AppendTo[PM`Private`data, "Settings" -> Association[]]]; object[PM`Private`data], HoldAll]]];

object /: Serialize[PM`Private`X_object] := SerializeHold[Initialize][Head[PM`Private`X], Serialize[PM`Private`X[[1]]]];

object /: ExportObject[PM`Private`file_, PM`Private`X_object] := Module[{PM`Private`Y}, PM`Private`Y \[LeftArrow] PM`Private`X; ClearCache[PM`Private`Y]; ClearPersistentCache[PM`Private`Y]; Export[PM`Private`file, Serialize[PM`Private`Y]]; PM`Private`file];

ImportObject[PM`Private`file_] := Deserialize[Import[PM`Private`file]];

object /: Equal[PM`Private`a__object] := Equal @@ KeyDrop[Flatten[{PM`Private`a}][[All,1]], {"Cache", "PersistentCache"}];

object /: MBCount[PM`Private`x_object] := Total[MBCount /@ Join[KeyDrop[PM`Private`x[[1]], {"Dimension", "Cache", "PersistentCache"}], Cache[PM`Private`x], PersistentCache[PM`Private`x]]];

object /: Identifier[PM`Private`x:object[PM`Objects`$object_]] := ToString[Unevaluated[PM`Objects`$object]];

object /: IDNumber[PM`Private`x:object[PM`Objects`$object_]] := With[{PM`Private`s = ToString[Unevaluated[PM`Objects`$object]]}, ToExpression[StringTake[PM`Private`s, {StringPosition[PM`Private`s, "$"][[-1,-1]] + 1, -1}]]];

object /: ObjectQ[object] := True;

object /: Settings[PM`Private`x:object[PM`Objects`$object_], args___] := PM`Objects`$object[["Settings",args]];

object /: SettingsKeys[PM`Private`x:object[PM`Objects`$object_]] := Keys[PM`Objects`$object[["Settings"]]];

object /: ClearSettings[PM`Private`x:object[PM`Objects`$object_]] := (PM`Objects`$object[["Settings"]] = Association[]; );

object /: ClearSettings[PM`Private`x:object[PM`Objects`$object_], All] := (PM`Objects`$object[["Settings"]] = Association[]; );

object /: ClearSettings[PM`Private`x:object[PM`Objects`$object_], PM`Private`s_String] := KeyDropFrom[PM`Objects`$object[["Settings"]], PM`Private`s];

object /: ClearSettings[PM`Private`x:object[PM`Objects`$object_], PM`Private`p_] := KeyDropFrom[PM`Objects`$object[["Settings"]], ToString[PM`Private`p, InputForm]];

object /: ClearSettings[PM`Private`x:object[PM`Objects`$object_], {PM`Private`s___String}] := KeyDropFrom[PM`Objects`$object[["Settings"]], {PM`Private`s}];

object /: ClearSettings[PM`Private`x:object[PM`Objects`$object_], {PM`Private`p__}] := KeyDropFrom[PM`Objects`$object[["Settings"]], Function[PM`Private`z, If[StringQ[PM`Private`z], PM`Private`z, ToString[PM`Private`z, InputForm]]] /@ Flatten[{PM`Private`p}]];

object /: SetSettings[PM`Private`x:object[PM`Objects`$object_], PM`Private`pos_, PM`Private`val_] := (PM`Objects`$object[["Settings",PM`Private`pos]] = PM`Private`val; );

profile /: ClearAllCache[PM`Private`x_profile, PM`Private`arg_:All] := (ClearCache[PM`Private`x, PM`Private`arg]; ClearPersistentCache[PM`Private`x, PM`Private`arg]; );

profile /: ClearAllCacheDependingOn[PM`Private`x_profile, PM`Private`s_] := ClearAllCache[PM`Private`x, VertexInComponent[CallGraph[$PM], ToString /@ Flatten[{PM`Private`s}]]];

profile /: Cache[PM`Private`x:profile[PM`Objects`$profile_], args___] := PM`Objects`$profile[["Cache",args]];

profile /: CacheKeys[PM`Private`x:profile[PM`Objects`$profile_]] := Keys[PM`Objects`$profile[["Cache"]]];

profile /: ClearCache[PM`Private`x:profile[PM`Objects`$profile_]] := (PM`Objects`$profile[["Cache"]] = Association[]; );

profile /: ClearCache[PM`Private`x:profile[PM`Objects`$profile_], All] := (PM`Objects`$profile[["Cache"]] = Association[]; );

profile /: ClearCache[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`s_String] := KeyDropFrom[PM`Objects`$profile[["Cache"]], PM`Private`s];

profile /: ClearCache[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`p_] := KeyDropFrom[PM`Objects`$profile[["Cache"]], ToString[PM`Private`p, InputForm]];

profile /: ClearCache[PM`Private`x:profile[PM`Objects`$profile_], {PM`Private`s___String}] := KeyDropFrom[PM`Objects`$profile[["Cache"]], {PM`Private`s}];

profile /: ClearCache[PM`Private`x:profile[PM`Objects`$profile_], {PM`Private`p__}] := KeyDropFrom[PM`Objects`$profile[["Cache"]], Function[PM`Private`z, If[StringQ[PM`Private`z], PM`Private`z, ToString[PM`Private`z, InputForm]]] /@ Flatten[{PM`Private`p}]];

profile /: ClearCacheDependingOn[PM`Private`x_profile, PM`Private`s_] := ClearCache[PM`Private`x, VertexInComponent[CallGraph[$PM], ToString /@ Flatten[{PM`Private`s}]]];

profile /: SetCache[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`pos_, PM`Private`val_] := (PM`Objects`$profile[["Cache",PM`Private`pos]] = PM`Private`val; );

profile /: PersistentCache[PM`Private`x:profile[PM`Objects`$profile_], args___] := PM`Objects`$profile[["PersistentCache",args]];

profile /: PersistentCacheKeys[PM`Private`x:profile[PM`Objects`$profile_]] := Keys[PM`Objects`$profile[["PersistentCache"]]];

profile /: ClearPersistentCache[PM`Private`x:profile[PM`Objects`$profile_]] := (PM`Objects`$profile[["PersistentCache"]] = Association[]; );

profile /: ClearPersistentCache[PM`Private`x:profile[PM`Objects`$profile_], All] := (PM`Objects`$profile[["PersistentCache"]] = Association[]; );

profile /: ClearPersistentCache[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`s_String] := KeyDropFrom[PM`Objects`$profile[["PersistentCache"]], PM`Private`s];

profile /: ClearPersistentCache[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`p_] := KeyDropFrom[PM`Objects`$profile[["PersistentCache"]], ToString[PM`Private`p, InputForm]];

profile /: ClearPersistentCache[PM`Private`x:profile[PM`Objects`$profile_], {PM`Private`s___String}] := KeyDropFrom[PM`Objects`$profile[["PersistentCache"]], {PM`Private`s}];

profile /: ClearPersistentCache[PM`Private`x:profile[PM`Objects`$profile_], {PM`Private`p__}] := KeyDropFrom[PM`Objects`$profile[["PersistentCache"]], Function[PM`Private`z, If[StringQ[PM`Private`z], PM`Private`z, ToString[PM`Private`z, InputForm]]] /@ Flatten[{PM`Private`p}]];

profile /: ClearPersistentCacheDependingOn[PM`Private`x_profile, PM`Private`s_] := ClearPersistentCache[PM`Private`x, VertexInComponent[CallGraph[$PM], ToString /@ Flatten[{PM`Private`s}]]];

profile /: SetPersistentCache[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`pos_, PM`Private`val_] := (PM`Objects`$profile[["PersistentCache",PM`Private`pos]] = PM`Private`val; );

SetAttributes[profile, HoldAll];

ToExpression[StringJoin["PM`Private`$", "profile", "Counter"], InputForm, Function[PM`Private`data, PM`Private`data = 0, HoldAll]];

Options[profile] = {"CacheActivated" -> True, "PersistentCacheActivated" -> True};

profile /: DeepCopy[PM`Private`x:profile[PM`Objects`$profile_]] := Initialize[profile, DeepCopy /@ PM`Objects`$profile];

profile /: (PM`Private`y_) \[LeftArrow] (PM`Private`x:profile[PM`Objects`$nprofile_]) := PM`Private`y = DeepCopy[PM`Private`x];

profile /: (PM`Private`x:profile[PM`Objects`$profile_])[[1,args__]] := PM`Objects`$profile[[args]];

profile /: (PM`Private`x:profile[PM`Objects`$profile_])[[PM`Private`s_String,args___]] := PM`Objects`$profile[[PM`Private`s,args]];

profile /: (PM`Private`x_profile)[PM`Private`s_String, args___] := PM`Private`x[[1]][[PM`Private`s,args]];

profile /: Initialize[profile, PM`Private`data0_Association] := With[{PM`Private`c = ToExpression[StringJoin["++PM`Private`$", "profile", "Counter"]]}, ToExpression[StringJoin["PM`Objects`$", "profile", "$", ToString[PM`Private`c]], InputForm, Function[PM`Private`data, SetAttributes[PM`Private`data, Temporary]; PM`Private`data = PM`Private`data0; If[ !KeyExistsQ[PM`Private`data, "Cache"], AppendTo[PM`Private`data, "Cache" -> Association[]]]; If[ !KeyExistsQ[PM`Private`data, "PersistentCache"], AppendTo[PM`Private`data, "PersistentCache" -> Association[]]]; If[ !KeyExistsQ[PM`Private`data, "Settings"], AppendTo[PM`Private`data, "Settings" -> Association[]]]; profile[PM`Private`data], HoldAll]]];

profile /: Serialize[PM`Private`X_profile] := SerializeHold[Initialize][Head[PM`Private`X], Serialize[PM`Private`X[[1]]]];

profile /: ExportObject[PM`Private`file_, PM`Private`X_profile] := Module[{PM`Private`Y}, PM`Private`Y \[LeftArrow] PM`Private`X; ClearCache[PM`Private`Y]; ClearPersistentCache[PM`Private`Y]; Export[PM`Private`file, Serialize[PM`Private`Y]]; PM`Private`file];

ImportObject[PM`Private`file_] := Deserialize[Import[PM`Private`file]];

profile /: Equal[PM`Private`a__profile] := Equal @@ KeyDrop[Flatten[{PM`Private`a}][[All,1]], {"Cache", "PersistentCache"}];

profile /: MBCount[PM`Private`x_profile] := Total[MBCount /@ Join[KeyDrop[PM`Private`x[[1]], {"Dimension", "Cache", "PersistentCache"}], Cache[PM`Private`x], PersistentCache[PM`Private`x]]];

profile /: Identifier[PM`Private`x:profile[PM`Objects`$profile_]] := ToString[Unevaluated[PM`Objects`$profile]];

profile /: IDNumber[PM`Private`x:profile[PM`Objects`$profile_]] := With[{PM`Private`s = ToString[Unevaluated[PM`Objects`$profile]]}, ToExpression[StringTake[PM`Private`s, {StringPosition[PM`Private`s, "$"][[-1,-1]] + 1, -1}]]];

profile /: ObjectQ[profile] := True;

profile /: Settings[PM`Private`x:profile[PM`Objects`$profile_], args___] := PM`Objects`$profile[["Settings",args]];

profile /: SettingsKeys[PM`Private`x:profile[PM`Objects`$profile_]] := Keys[PM`Objects`$profile[["Settings"]]];

profile /: ClearSettings[PM`Private`x:profile[PM`Objects`$profile_]] := (PM`Objects`$profile[["Settings"]] = Association[]; );

profile /: ClearSettings[PM`Private`x:profile[PM`Objects`$profile_], All] := (PM`Objects`$profile[["Settings"]] = Association[]; );

profile /: ClearSettings[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`s_String] := KeyDropFrom[PM`Objects`$profile[["Settings"]], PM`Private`s];

profile /: ClearSettings[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`p_] := KeyDropFrom[PM`Objects`$profile[["Settings"]], ToString[PM`Private`p, InputForm]];

profile /: ClearSettings[PM`Private`x:profile[PM`Objects`$profile_], {PM`Private`s___String}] := KeyDropFrom[PM`Objects`$profile[["Settings"]], {PM`Private`s}];

profile /: ClearSettings[PM`Private`x:profile[PM`Objects`$profile_], {PM`Private`p__}] := KeyDropFrom[PM`Objects`$profile[["Settings"]], Function[PM`Private`z, If[StringQ[PM`Private`z], PM`Private`z, ToString[PM`Private`z, InputForm]]] /@ Flatten[{PM`Private`p}]];

profile /: SetSettings[PM`Private`x:profile[PM`Objects`$profile_], PM`Private`pos_, PM`Private`val_] := (PM`Objects`$profile[["Settings",PM`Private`pos]] = PM`Private`val; );

(** Verbatims **)

$EigenIncludeDirectory = FileNameJoin[{SourcePath[$PM], "PMTools", "eigen3", "Eigen"}];

$OpenMPIncludeDirectory = FileNameJoin[{SourcePath[$PM], "PMTools", "openmp"}];

$CountingData = Association[];

SetAttributes[CountingReap, HoldAll];

CountingReap[] := Module[{toc0, oldtag0}, $CountingData = Association[]];

CountingReap[code0_] := Module[{data, toc0, oldtag0}, $CountingData = Association[]; TimingSow = Function[{code, tag, args, type}, If[KeyExistsQ[$CountingData, {tag, args, type}], $CountingData[{tag, args, type}]++, $CountingData[{tag, args, type}] = 1]; ReleaseHold[code], HoldAll]; data = ReleaseHold[code0]; TimingSow = Function[Null, #1, HoldAll]; $CountingData];

SetAttributes[CreateFormat, HoldAll];

CreateFormat[obj_, visible_, hidden_, OptionsPattern[{"PrintCode" -> False, "ProcessCode" -> True, "EvaluationCondition" -> Function[z, AssociationQ[z[[1]]]]}]] := Module[{object, $object, item, s}, object = ToString[obj]; $object = StringJoin["$", object]; item[a_Association] := StringJoin["BoxForm`MakeSummaryItem[{", ToString[a[["Caption"]], InputForm], ", ", StringJoin["With[{data=", ToString[a[["Function"]]], "[x]},If[Head[data] === ", ToString[a[["Function"]]], ",Missing[],data]\n]"], "}, StandardForm]"]; item[a_] := ToString[a, InputForm]; s = StringJoin[object, " /: MakeBoxes[x_", object, ", StandardForm] := BoxForm`ArrangeSummaryBox[", object, ", \"\", ", "\"ID:  \"<>ToString[IDNumber[x]]", ", ", StringJoin["{", Riffle[Function[x, StringJoin["\n\t{\n\t\t", Riffle[x, ",\n\t\t"], "\n\t}"]] /@ Map[item, visible, {2}], ","], "\n}\n"], ", ", StringJoin["{", Riffle[Function[x, StringJoin["\n\t{\n\t\t", Riffle[x, ",\n\t\t"], "\n\t}"]] /@ Map[item, hidden, {2}], ","], "\n}\n"], ", StandardForm, \"Interpretable\" -> False] /; (", ToString[OptionValue["EvaluationCondition"]], "[x])"]; If[OptionValue["PrintCode"], Print[s]]; If[OptionValue["ProcessCode"], ToExpression[s]]; ];

ToExpression["TagSetDelayed[object, Format[Pattern[P, Blank[object]]], Quiet@SequenceForm[\"object[\", List[], \"]\"]]"];

object /: Init[object] := Initialize[object, Association["Cache" -> Association[], "PersistentCache" -> Association[]]];

ClearAll[Inherit];

SetAttributes[Inherit, HoldAll];

Inherit /: (newobject_ = Inherit[oldobject_, overwrite_:False]) := TimingSow[Module[{minor, major, a, b, duplicates, bslist, bpos}, If[overwrite, minor = newobject; major = oldobject; , major = newobject; minor = oldobject; ]; Do[With[{f = fun}, a = f[Evaluate[major]] /. {major -> newobject}; b = f[Evaluate[minor]] /. {minor -> newobject}; If[f === Attributes, a = List /@ a; b = List /@ b; ]; duplicates = Intersection[a[[All,1]], b[[All,1]]]; bslist = ToString /@ b[[All,1]]; bpos = Flatten[Table[Position[bslist, z], {z, ToString /@ duplicates}]]; f[newobject] = Join[a, b[[Complement[Range[Length[b]], bpos]]]]; ], {fun, {OwnValues, DownValues, Messages, Options, Attributes, UpValues, FormatValues}}]; ], "Inherit", {newobject}, oldobject];

profile /: Init[profile, result_, data_Association, OptionsPattern[{}]] := Initialize[profile, Association["Result" -> result, "Data" -> data]];

CreateFormat[profile, {{Association["Caption" -> "Calls: ", "Function" -> NumberOfCalls], Association["Caption" -> "TotalTime: ", "Function" -> TotalTime]}}, {}];

DebuggingReap::abort = "Execution aborted.";

DebuggingReap::error = "Error found during execution of a call from `1` to `2` for `3` with arguments `4`.";

SetAttributes[DebuggingReap, HoldAll];

DebuggingReap[code0_] := Module[{data, oldtag0}, oldtag0 = None; $TimingStack = {{0, TimingReap}}; $TimingStackCounter = 0; $TimingCallCounter = 0; $TimingInitial = AbsoluteTime[]; $TimingData = Internal`Bag[{}]; TimingSow = Function[{code, tag, vars, type}, Block[{result, from, cc, cs}, from = $TimingStack[[1]]; cs = ++$TimingStackCounter; cc = ++$TimingCallCounter; $TimingStack = {{cc, tag}, $TimingStack}; result = Check[code, Message[DebuggingReap::error, Style[ToString[from[[2]]], Bold], Style[ToString[tag], Bold], Style[ToString[type], Bold], Style[ToString[vars], Bold]]; Abort[]]; $TimingStack = $TimingStack[[2]]; --$TimingStackCounter; result], HoldAll]; PrintTemporary["TimingStack: ", Dynamic[$TimingStack[[1]]]]; data = CheckAbort[ReleaseHold[code0], Message[DebuggingReap::abort]; ]; TimingSow = Function[Null, #1, HoldAll]; data];

SetAttributes[TimingReap, HoldAll];

TimingReap[] := Module[{toc0, oldtag0}, oldtag0 = None; $TimingStack = {{0, TimingReap}}; $TimingStackCounter = 0; $TimingCallCounter = 0; $TimingInitial = AbsoluteTime[]; $TimingData = Internal`Bag[{}]; toc0 = AbsoluteTime[]; TimingSow = Function[Null, #1, HoldAll]; Init[profile, Null, KeySort[Prepend[Association[Internal`BagPart[$TimingData, All]], Association[0 -> Association["Tag" -> TimingReap, "Type" -> None, "Arguments" -> {}, "Depth" -> 0, "Tic" -> 0., "Toc" -> toc0 - $TimingInitial, "From" -> {-1, "User"}]]]]]];

TimingReap[code0_] := Module[{data, toc0, oldtag0}, oldtag0 = None; $TimingStack = {{0, TimingReap}}; $TimingStackCounter = 0; $TimingCallCounter = 0; $TimingInitial = AbsoluteTime[]; $TimingData = Internal`Bag[{}]; TimingSow = Function[{code, tag, args, type}, Block[{tic, toc, result, cs, cc, from}, Internal`WithLocalSettings[from = $TimingStack[[1]]; cs = ++$TimingStackCounter; cc = ++$TimingCallCounter; $TimingStack = {{cc, tag}, $TimingStack}; tic = AbsoluteTime[]; , result = ReleaseHold[code]; , toc = AbsoluteTime[]; $TimingStack = $TimingStack[[2]]; Internal`StuffBag[$TimingData, cc -> Association["Tag" -> tag, "Type" -> type, "Arguments" -> args, "Depth" -> cs, "Tic" -> tic - $TimingInitial, "Toc" -> toc - $TimingInitial, "From" -> from]]; --$TimingStackCounter; ]; result], HoldAll]; data = ReleaseHold[code0]; toc0 = AbsoluteTime[]; TimingSow = Function[Null, #1, HoldAll]; Init[profile, data, KeySort[Prepend[Association[Internal`BagPart[$TimingData, All]], Association[0 -> Association["Tag" -> TimingReap, "Type" -> None, "Depth" -> 0, "Arguments" -> {}, "Tic" -> 0., "Toc" -> toc0 - $TimingInitial, "From" -> {-1, "User"}]]]]]];

(** CompiledFunctions **)

(** PackageFunctions **)

CallGraph[Sequence[]] := TimingSow[CallGraph[$PM], "CallGraph", {}, None];

CallsTo[Sequence[s_, depth_:2]] := TimingSow[CallsTo[$PM, s, depth], "CallsTo", {"s_", "depth_:2"}, None];

CallsFrom[Sequence[s_, depth_:2]] := TimingSow[CallsFrom[$PM, s, depth], "CallsFrom", {"s_", "depth_:2"}, None];

CallsWith[Sequence[s_, depth_:2]] := TimingSow[CallsWith[$PM, s, depth], "CallsWith", {"s_", "depth_:2"}, None];

packagemanager /: CallsTo[M:packagemanager[$packagemanager_], Sequence[s_, depth_:2]] := TimingSow[Module[{G, list, data}, data = ToString /@ Flatten[{s}]; G = CallGraph[M]; list = VertexInComponent[G, data, depth]; Graph[Subgraph[G, list], Options[G]]], "CallsTo", {"s_", "depth_:2"}, packagemanager];

packagemanager /: CallsWith[M:packagemanager[$packagemanager_], Sequence[s_, depth_:2]] := TimingSow[Module[{G, list, data}, data = ToString /@ Flatten[{s}]; G = CallGraph[M]; list = Join[VertexInComponent[G, data, depth], VertexOutComponent[G, data, depth]]; Graph[Subgraph[G, list], Options[G]]], "CallsWith", {"s_", "depth_:2"}, packagemanager];

packagemanager /: CallsFrom[M:packagemanager[$packagemanager_], Sequence[s_, depth_:2]] := TimingSow[Module[{G, list, data}, data = ToString /@ Flatten[{s}]; G = CallGraph[M]; list = VertexOutComponent[G, data, depth]; Graph[Subgraph[G, list], Options[G]]], "CallsFrom", {"s_", "depth_:2"}, packagemanager];

FindNotebooks[path_String] := TimingSow[FileNames[FileNameJoin[{path, "*.nb"}]], "FindNotebooks", {"path_String"}, None];

DeclutterNotebooks[file_String] := TimingSow[Module[{data, nbfiles, openednbfiles, closednbfiles, nb, pos, nb0, nb1, mem0, mem1, c}, data = Association["Opened" -> Association[], "Closed" -> Association[]]; If[DirectoryQ[file], nbfiles = FindNotebooks[file]; , nbfiles = FileNames[file]]; openednbfiles = Intersection[NotebookFileName /@ Most[Notebooks[]], nbfiles]; closednbfiles = Complement[nbfiles, openednbfiles]; Do[nb = nbfile[[1]]; NotebookFind[nb, "Output", All, CellStyle]; If[Length[SelectedCells[nb]] > 0, With[{name = NotebookFileName[nb]}, Print[name]; mem0 = Quantity[FileByteCount[name], "Byte"]; NotebookDelete[nb, AutoScroll -> Delete]; NotebookSave[nb]; mem1 = Quantity[FileByteCount[name], "Byte"]; AppendTo[data[["Opened"]], name -> mem0 - mem1]]; ]; , {nbfile, Notebooks /@ openednbfiles}]; Do[nb0 = Get[nbfile]; pos = Position[nb0, _CellGroupData]; If[Length[pos] > 0, c = 0; nb1 = (c++; MapAt[Function[x, Sequence @@ x[[1,1]]], nb0, pos]); If[c > 0, mem0 = Quantity[FileByteCount[nbfile], "Byte"]; Put[nb1, nbfile]; mem1 = Quantity[FileByteCount[nbfile], "Byte"]; AppendTo[data[["Closed"]], nbfile -> mem0 - mem1]; ]], {nbfile, closednbfiles}]; data], "DeclutterNotebooks", {"file_String"}, None];

DeclutterNotebooks[files_List] := TimingSow[Merge[DeclutterNotebooks /@ Flatten[files], Function[x, Merge[x, First]]], "DeclutterNotebooks", {"files_List"}, None];

packagemanager /: PackageSourceDeclutterNotebooks[M:packagemanager[$packagemanager_]] := TimingSow[PackageSourceBackup[M]; DeclutterNotebooks[Subdirectories[getp[M], 12]], "PackageSourceDeclutterNotebooks", {}, packagemanager];

fixBSplineFunction[Sequence[]] := TimingSow[Module[{\[Gamma]}, \[Gamma] = BSplineFunction[RandomReal[{-1, 1}, {6, 3}]]; Quiet[ReleaseHold[MakeExpression[Activate[Inactive[ToString][Inactive[Definition][ElisionsDump`makeSplineBoxes], InputForm]]] /. {{{BoxForm`t, 0, 1}} -> {{BoxForm`t, ElisionsDump`spline[[2,1,1]], ElisionsDump`spline[[2,1,2]]}}, {{BoxForm`s, 0, 1}, {BoxForm`t, 0, 1}} -> {{BoxForm`s, ElisionsDump`spline[[2,1,1]], ElisionsDump`spline[[2,1,2]]}, {BoxForm`t, ElisionsDump`spline[[2,2,1]], ElisionsDump`spline[[2,2,2]]}}}]]], "fixBSplineFunction", {}, None];

MemoryStatistics[M_] := TimingSow[Module[{data, k, v, mbcount, fullMB, relmbcount}, data = Join[KeyDrop[M[[1]], {"Dimension", "Cache", "PersistentCache"}], KeyDrop[Cache[M], {}], KeyDrop[PersistentCache[M], {}]]; k = Keys[data]; v = Values[data]; mbcount = MBCount /@ v; fullMB = Total[mbcount]; relmbcount = Quantity[100.*(mbcount/fullMB), "Percent"]; Dataset[Reverse[SortBy[Association[Thread[k -> Apply[Association, Transpose[{Thread["MB" -> mbcount], Thread["%" -> relmbcount]}], {1}]]], First]]]], "MemoryStatistics", {"M_"}, None];

MemoryChart[M_] := TimingSow[With[{data = MemoryStatistics[M]}, Show[PieChart3D[Function[x, Tooltip[x["MB"], Row[{NumberForm[x[["MB"]], {6, 2, 10}], Spacer[10], "(", NumberForm[x[["%"]], {6, 2, 10}], ")"}]]] /@ Normal[data[All, {"MB", "%"}]], ColorFunction -> Function[{x}, ColorData["DarkRainbow"][x]], SectorOrigin -> {Automatic, 1}, ChartLabels -> Placed[Keys[Normal[data]], "RadialCallout"], ChartElementFunction -> "ProfileSector3D"], PlotRange -> All, ImageSize -> Large]], "MemoryChart", {"M_"}, None];

OpenSource::morethanone = "More than one source found. Please specify which one to open by calling OpenSource[M,`1`,n].";

packagemanager /: OpenSource[M:packagemanager[$packagemanager_], fun_] := TimingSow[Module[{source}, source = Source[M, fun]; If[Length[source] > 1, Message[OpenSource::morethanone, fun]; Grid[MapIndexed[Function[{s, i}, {i[[1]], "\[Rule]", Button[Style[s, "Hyperlink", FontFamily -> "Courier"], Module[{nb}, nb = NotebookOpen[s[["File"]]]; SelectionMove[nb, Before, Notebook]; SelectionMove[nb, Next, Cell, s[["Cell"]]]], Alignment -> Left, Appearance -> None]}], source], Alignment -> {{Right, Center, Left}}], OpenSource[M, fun, 1]]], "OpenSource", {"fun_"}, packagemanager];

packagemanager /: OpenSource[M:packagemanager[$packagemanager_], Sequence[fun_, i_]] := TimingSow[Module[{source, nb}, source = Source[M, fun][[i]]; nb = NotebookOpen[source[["File"]]]; SelectionMove[nb, Before, Notebook]; SelectionMove[nb, Next, Cell, source[["Cell"]]]; ], "OpenSource", {"fun_", "i_"}, packagemanager];

OpenSource[fun_] := TimingSow[OpenSource[$PM, fun], "OpenSource", {"fun_"}, None];

OpenSource[Sequence[fun_, i_]] := TimingSow[OpenSource[$PM, fun, i], "OpenSource", {"fun_", "i_"}, None];

packagemanager /: PackageSourceBackup[M:packagemanager[$packagemanager_]] := TimingSow[CopyDirectory[getp[M], FileNameJoin[{$HomeDirectory, StringJoin[FileBaseName[getp[M]], "_Backup_", DateString[{"Year", "-", "Month", "-", "Day", "__", "Hour", "_", "Minute", "_", "Second"}]]}]], "PackageSourceBackup", {}, packagemanager];

packagemanager /: PackageSourceCopyRequired[PM:packagemanager[$packagemanager_], Sequence[loadlist_, destinationpath_]] := TimingSow[Module[{G, path, destination, SubfolderWhiteList, SubfolderBlackList}, G = LoadGraph[PM, loadlist]; SubfolderWhiteList = {"Geometries", "KnotData", "LibrarySources", "Examples"}; SubfolderBlackList = {"Development", "Hide"}; path = PM[[1,"SourcePath"]]; destination = FileNameJoin[{destinationpath, "PackageSources"}]; If[FileExistsQ[destination], DeleteDirectory[destination, DeleteContents -> True]; ]; CreateDirectory[destination]; Do[CopyDirectory[FileNameJoin[{path, dir}], FileNameJoin[{destination, dir}]]; DeleteDirectory[Complement[Select[FileNames[FileNameJoin[{FileNameJoin[{destination, dir}], "*"}]], DirectoryQ], (FileNameJoin[{destination, dir, #1}] & ) /@ SubfolderWhiteList], DeleteContents -> True]; , {dir, VertexList[G]}]; CopyDirectory[FileNameJoin[{path, "PM", "LTemplate"}], FileNameJoin[{destination, "PM", "LTemplate"}]]; CopyFile[FileNameJoin[{path, "InstallPackageManager.nb"}], FileNameJoin[{destination, "InstallPackageManager.nb"}]]; CopyDirectory[FileNameJoin[{path, "PMTools", "eigen3"}], FileNameJoin[{destination, "PMTools", "eigen3"}]]; CopyDirectory[FileNameJoin[{path, "PMTools", "openmp"}], FileNameJoin[{destination, "PMTools", "openmp"}]]; destination], "PackageSourceCopyRequired", {"loadlist_", "destinationpath_"}, packagemanager];

packagemanager /: PackageSourceFindCommand[M:packagemanager[$packagemanager_], Sequence[command_, depth_:1]] := TimingSow[PackageSourceFindCommand[M, ToString[command], depth], "PackageSourceFindCommand", {"command_", "depth_:1"}, packagemanager];

packagemanager /: PackageSourceFindCommand[M:packagemanager[$packagemanager_], Sequence[s_String, depth_:1]] := TimingSow[Module[{repo, dirs, nbfiles, data, nb, c, r, pos, openednbfiles, closednbfiles, nb0, nbexpr}, dirs = Subdirectories[getp[M], depth]; nb0 = SelectedNotebook[]; nbfiles = Complement[Union @@ FindNotebooks /@ dirs, Flatten[{NotebookFileName[nb0]}]]; data = Association["Skipped" -> NotebookFileName[nb0], "Opened" -> Association[], "Closed" -> Association[]]; openednbfiles = Intersection[NotebookFileName /@ Most[Notebooks[]], nbfiles]; closednbfiles = Complement[nbfiles, openednbfiles]; Do[nb = Notebooks[file][[1]]; NotebookSave[nb]; NotebookClose[nb]; nbexpr = Get[file]; r = Position[nbexpr, s]; If[Length[r] > 0, AppendTo[data[["Opened"]], file -> r]]; NotebookOpen[file]; , {file, openednbfiles}]; Do[nbexpr = Get[file]; r = Position[nbexpr, s]; If[Length[r] > 0, AppendTo[data[["Closed"]], file -> r]]; , {file, closednbfiles}]; SetSelectedNotebook[nb0]; data], "PackageSourceFindCommand", {"s_String", "depth_:1"}, packagemanager];

packagemanager /: PackageSourceReplaceCommand[M:packagemanager[$packagemanager_], Sequence[{r0__Rule}, depth_:1]] := TimingSow[Module[{repo, dirs, nbfiles, data, nb, c, pos, openednbfiles, closednbfiles, nb0, nbexpr}, PackageSourceBackup[M]; repo = Map[ToString, {r0}, {2}]; dirs = Subdirectories[getp[M], depth]; nb0 = SelectedNotebook[]; nbfiles = Complement[Union @@ FindNotebooks /@ dirs, Flatten[{NotebookFileName[nb0]}]]; data = Association["Opened" -> Association[], "Closed" -> Association[]]; openednbfiles = Intersection[NotebookFileName /@ Most[Notebooks[]], nbfiles]; closednbfiles = Complement[nbfiles, openednbfiles]; Do[nb = Notebooks[file][[1]]; NotebookSave[nb]; NotebookClose[nb]; nbexpr = Get[file]; c = 0; Do[pos = Position[nbexpr, r[[1]]]; If[Length[pos] > 0, c += Length[pos]; nbexpr = MapAt[Function[x, r[[2]]], nbexpr, pos]; ], {r, repo}]; If[c > 0, AppendTo[data[["Opened"]], file -> c]; Put[nbexpr, file]; ]; NotebookOpen[file]; , {file, openednbfiles}]; SetSelectedNotebook[nb0]; Do[nbexpr = Get[file]; c = 0; Do[pos = Position[nbexpr, r[[1]]]; If[Length[pos] > 0, c += Length[pos]; nbexpr = MapAt[Function[x, r[[2]]], nbexpr, pos]; ], {r, repo}]; If[c > 0, AppendTo[data[["Closed"]], file -> c]; Put[nbexpr, file]; ]; , {file, closednbfiles}]; Print["Number of altered files: ", Length[Keys[data]], ". ", "Total number of changes: ", Total[data], ". "]; data], "PackageSourceReplaceCommand", {"{r0__Rule}", "depth_:1"}, packagemanager];

profile /: NumberOfCalls[p:profile[$profile_]] := TimingSow[Length[p[[1]][["Data"]]], "NumberOfCalls", {}, profile];

profile /: Data[p:profile[$profile_]] := If[KeyExistsQ[$profile[["Cache"]], "Data"], $profile[["Cache","Data"]], TimingSow[$profile[["Cache","Data"]] = Module[{data, A, times, owntimes}, data = p[[1,"Data"]]; times = Values[data[[All,"Toc"]] - data[[All,"Tic"]]]; A = SparseArray[Apply[List, Normal[data[[2 ;; All,"From",1]]], {1}] + 1 -> 1., Length[times]]; owntimes = times - times . A; data[[All,"Eigentime"]] = owntimes; data], "Data", {}, profile]];

profile /: TotalTime[p:profile[$profile_]] := If[KeyExistsQ[$profile[["Cache"]], "TotalTime"], $profile[["Cache","TotalTime"]], TimingSow[$profile[["Cache","TotalTime"]] = p[[1,"Data",Key[0],"Toc"]] - p[[1,"Data",Key[0],"Tic"]], "TotalTime", {}, profile]];

profile /: Eigentimes[p:profile[$profile_]] := TimingSow[Data[p][[All,{"Tag", "Eigentime"}]], "Eigentimes", {}, profile];

profile /: EigentimeTable[p:profile[$profile_], threshold_:Quantity[5., "Percent"]] := TimingSow[Module[{tot, data, a, b, i}, tot = TotalTime[p]; data = Association[KeyValueMap[Function[{key, val}, key -> Association["Absolute Eigentime" -> Quantity[1000*val, "Milliseconds"], "Relative Eigentime" -> Quantity[100*(val/tot), "Percent"]]], ReverseSort[Merge[Apply[Rule, Values /@ Values[Eigentimes[p]], {1}], Total]]]]; a = Accumulate[Reverse[Values[data[[All,"Relative Eigentime"]]]]]; b = Accumulate[Reverse[Values[data[[All,"Absolute Eigentime"]]]]]; i = FirstPosition[(#1 > threshold & ) /@ Accumulate[Reverse[Values[data[[All,"Relative Eigentime"]]]]], True, 1][[1]]; Dataset[Append[data[[1 ;; -i]], "Others" -> Association["Absolute Eigentime" -> b[[i - 1]], "Relative Eigentime" -> a[[i - 1]]]]]], "EigentimeTable", {"threshold_:Quantity[5., \"Percent\"]"}, profile];

profile /: EigentimeChart[p:profile[$profile_], timetable_Dataset] := TimingSow[With[{data = Normal[timetable]}, Show[PieChart3D[Function[x, Tooltip[x["Absolute Eigentime"], Row[{NumberForm[x[["Absolute Eigentime"]], {6, 2, 10}], Spacer[10], "(", NumberForm[x[["Relative Eigentime"]], {6, 2, 10}], ")"}]]] /@ data, ColorFunction -> Function[{x}, ColorData["DarkRainbow"][x]], SectorOrigin -> {Automatic, 1}, ChartLabels -> Placed[Keys[data], "RadialCallout"], ChartElementFunction -> "ProfileSector3D"], PlotRange -> All, ImageSize -> Large]], "EigentimeChart", {"timetable_Dataset"}, profile];

profile /: EigentimeChart[p:profile[$profile_], threshold_:Quantity[5., "Percent"]] := TimingSow[With[{data = Normal[EigentimeTable[p, threshold]]}, Show[PieChart3D[Function[x, Tooltip[x["Absolute Eigentime"], Row[{NumberForm[x[["Absolute Eigentime"]], {6, 2, 10}], Spacer[10], "(", NumberForm[x[["Relative Eigentime"]], {6, 2, 10}], ")"}]]] /@ data, ColorFunction -> Function[{x}, ColorData["DarkRainbow"][x]], SectorOrigin -> {Automatic, 1}, ChartLabels -> Placed[Keys[data], "RadialCallout"], ChartElementFunction -> "ProfileSector3D"], PlotRange -> All, ImageSize -> Large]], "EigentimeChart", {"threshold_:Quantity[5., \"Percent\"]"}, profile];

profile /: SunburstLeaves[p:profile[$profile_]] := If[KeyExistsQ[$profile[["Cache"]], "SunburstLeaves"], $profile[["Cache","SunburstLeaves"]], TimingSow[$profile[["Cache","SunburstLeaves"]] = GroupBy[Normal[Data[p][[All,"From",1]]], Last -> First], "SunburstLeaves", {}, profile]];

profile /: SunburstData[p:profile[$profile_], OptionsPattern[{"Depth" -> Infinity, "StartWith" -> 0}]] := TimingSow[Module[{data, leaves, layernumber, factor, layerlist, firstinlayer, restoflayer, firstannuli, restannuli, r1, r2, datalist, t\[Alpha], t\[Omega], \[Alpha]0, r0, i0, depth, depth0, annuli, radii, firstcolfun, restcolfun, firstedgeform, restedgeform, root, t0, firstannulusfun, restannulusfun, restdatalists, centerdisk, totaltime}, data = Data[p]; leaves = SunburstLeaves[p]; \[Alpha]0 = Pi/2; r0 = 0.; i0 = OptionValue["StartWith"]; root = data[i0]; t0 = root[["Tic"]]; depth0 = root[["Depth"]]; totaltime = root[["Toc"]] - root[["Tic"]]; depth = Min[Max[data[[All,"Depth"]]], OptionValue["Depth"]]; radii = N[Range[depth + r0 + 1]/(depth + 1)]; restcolfun = ColorData["DarkRainbow"]; restedgeform = EdgeForm[{Black, Thin}]; layernumber = 0; factor = 2*(Pi/totaltime); layerlist = {i0}; centerdisk = {FaceForm[restcolfun[layernumber/depth]], restedgeform, {{Disk[{0., 0.}, 1.001*radii[[1]]], StringJoin["(", ToString[i0], ") ", ToString[root[["Tag"]]], " Time = ", ToString[totaltime], " s  (100.0 %)"]}}}; restannulusfun = Function[x, If[x[[3,2]] - x[[3,1]] > 10^(-4), {Annulus @@ x[[1 ;; 3]], StringJoin["(", ToString[x[[-3]]], ") ", ToString[x[[-2]]], " Time = ", ToString[x[[-1]]], " s  (", ToString[(x[[-1]]/totaltime)*100.], " %)"]}, Nothing]]; Prepend[Table[(layernumber++; firstinlayer = layerlist; restoflayer = leaves /@ layerlist; r1 = radii[[layernumber]]; r2 = radii[[layernumber + 1]]; restdatalists = Function[x, If[Head[x] === Missing, Missing[], data[[Key /@ x]]]] /@ restoflayer; layerlist = Flatten[DeleteMissing[restoflayer]]; datalist = Join @@ DeleteMissing[restdatalists]; ); t\[Alpha] = Values[datalist[[All,"Tic"]]]; t\[Omega] = Values[datalist[[All,"Toc"]]]; annuli = Transpose[{ConstantArray[{0., 0.}, Length[datalist]], ConstantArray[{r1, r2}, Length[datalist]], Transpose[{\[Alpha]0 - factor*(t\[Omega] - t0), \[Alpha]0 - factor*(t\[Alpha] - t0)}], Keys[datalist], Values[datalist[[All,"Tag"]]], t\[Omega] - t\[Alpha]}]; restannuli = {FaceForm[restcolfun[layernumber/depth]], restedgeform, restannulusfun /@ annuli}; {restannuli}, {i, depth0, depth - 1}], {centerdisk}]], "SunburstData", {"OptionsPattern[{\"Depth\" -> Infinity, \"StartWith\" -> 0}]"}, profile];

profile /: Sunburst[p:profile[$profile_], OptionsPattern[{"Tooltips" -> True, "Depth" -> Infinity, "StartWith" -> 0}]] := TimingSow[Module[{fun, depth, data}, data = SunburstData[p, "Depth" -> OptionValue["Depth"], "StartWith" -> OptionValue["StartWith"]]; depth = Min[Length[data], OptionValue["Depth"]]; fun = If[OptionValue["Tooltips"], Function[x, {x[[1,1 ;; 2]], Apply[Tooltip, x[[1,3]], {1}]}], Function[x, {x[[1,1 ;; 2]], x[[1,3,1]]}]]; Graphics[Flatten[fun /@ data[[1 ;; depth]]]]], "Sunburst", {"OptionsPattern[{\"Tooltips\" -> True, \"Depth\" -> Infinity, \"StartWith\" -> 0}]"}, profile];

profile /: Sunburst[p:profile[$profile_], Sequence[i_Integer, OptionsPattern[{"Tooltips" -> True, "Depth" -> Infinity}]]] := TimingSow[Sunburst[p, "StartWith" -> i, "Tooltips" -> OptionValue["Tooltips"], "Depth" -> OptionValue["Depth"]], "Sunburst", {"i_Integer", "OptionsPattern[{\"Tooltips\" -> True, \"Depth\" -> Infinity}]"}, profile];



End[];

