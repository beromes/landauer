module top(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13);
  input x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26;
  output y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13;
  wire n28, n29, n30, n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104, n105, n106, n107, n108, n109, n110, n111, n112, n113, n114, n115, n116, n117, n118, n119, n120, n121, n122, n123, n124, n125, n126, n127, n128, n129, n130, n131, n132, n133, n134, n135, n136, n137, n138, n139, n140, n141, n142, n143, n144, n145, n146, n147, n148, n149, n150, n151, n152, n153, n154, n155, n156, n157, n158, n159, n160, n161, n162, n163, n164, n165, n166, n167, n168, n169, n170, n171, n172, n173, n174, n175, n176, n177, n178, n179, n180;
  assign n28 = ~x0 & x13;
  assign n29 = x0 & ~x13;
  assign n30 = ~n28 & ~n29;
  assign n31 = x26 & n30;
  assign n32 = ~x26 & ~n30;
  assign n33 = ~n31 & ~n32;
  assign n34 = ~x1 & x14;
  assign n35 = x1 & ~x14;
  assign n36 = ~n34 & ~n35;
  assign n37 = x0 & x13;
  assign n38 = x26 & ~n30;
  assign n39 = ~n37 & ~n38;
  assign n40 = n36 & ~n39;
  assign n41 = ~n36 & n39;
  assign n42 = ~n40 & ~n41;
  assign n43 = ~x2 & x15;
  assign n44 = x2 & ~x15;
  assign n45 = ~n43 & ~n44;
  assign n48 = ~n36 & n38;
  assign n46 = ~n36 & n37;
  assign n47 = x1 & x14;
  assign n49 = ~n46 & ~n47;
  assign n50 = ~n48 & n49;
  assign n51 = n45 & ~n50;
  assign n52 = ~n45 & n50;
  assign n53 = ~n51 & ~n52;
  assign n54 = ~x3 & x16;
  assign n55 = x3 & ~x16;
  assign n56 = ~n54 & ~n55;
  assign n60 = ~n45 & n48;
  assign n58 = ~n45 & n46;
  assign n57 = ~n45 & n47;
  assign n59 = x2 & x15;
  assign n61 = ~n57 & ~n59;
  assign n62 = ~n58 & n61;
  assign n63 = ~n60 & n62;
  assign n64 = n56 & ~n63;
  assign n65 = ~n56 & n63;
  assign n66 = ~n64 & ~n65;
  assign n67 = ~x4 & x17;
  assign n68 = x4 & ~x17;
  assign n69 = ~n67 & ~n68;
  assign n74 = ~n56 & n60;
  assign n72 = ~n56 & n58;
  assign n71 = ~n56 & n57;
  assign n70 = ~n56 & n59;
  assign n73 = x3 & x16;
  assign n75 = ~n70 & ~n73;
  assign n76 = ~n71 & n75;
  assign n77 = ~n72 & n76;
  assign n78 = ~n74 & n77;
  assign n79 = n69 & ~n78;
  assign n80 = ~n69 & n78;
  assign n81 = ~n79 & ~n80;
  assign n82 = ~x5 & x18;
  assign n83 = x5 & ~x18;
  assign n84 = ~n82 & ~n83;
  assign n85 = x4 & x17;
  assign n86 = ~n69 & ~n78;
  assign n87 = ~n85 & ~n86;
  assign n88 = n84 & ~n87;
  assign n89 = ~n84 & n87;
  assign n90 = ~n88 & ~n89;
  assign n91 = ~x6 & x19;
  assign n92 = x6 & ~x19;
  assign n93 = ~n91 & ~n92;
  assign n96 = ~n84 & n86;
  assign n94 = ~n84 & n85;
  assign n95 = x5 & x18;
  assign n97 = ~n94 & ~n95;
  assign n98 = ~n96 & n97;
  assign n99 = n93 & ~n98;
  assign n100 = ~n93 & n98;
  assign n101 = ~n99 & ~n100;
  assign n102 = ~x7 & x20;
  assign n103 = x7 & ~x20;
  assign n104 = ~n102 & ~n103;
  assign n108 = ~n93 & n96;
  assign n106 = ~n93 & n94;
  assign n105 = ~n93 & n95;
  assign n107 = x6 & x19;
  assign n109 = ~n105 & ~n107;
  assign n110 = ~n106 & n109;
  assign n111 = ~n108 & n110;
  assign n112 = n104 & ~n111;
  assign n113 = ~n104 & n111;
  assign n114 = ~n112 & ~n113;
  assign n115 = ~x8 & x21;
  assign n116 = x8 & ~x21;
  assign n117 = ~n115 & ~n116;
  assign n122 = ~n104 & n108;
  assign n120 = ~n104 & n106;
  assign n119 = ~n104 & n105;
  assign n118 = ~n104 & n107;
  assign n121 = x7 & x20;
  assign n123 = ~n118 & ~n121;
  assign n124 = ~n119 & n123;
  assign n125 = ~n120 & n124;
  assign n126 = ~n122 & n125;
  assign n127 = n117 & ~n126;
  assign n128 = ~n117 & n126;
  assign n129 = ~n127 & ~n128;
  assign n130 = ~x9 & x22;
  assign n131 = x9 & ~x22;
  assign n132 = ~n130 & ~n131;
  assign n133 = x8 & x21;
  assign n134 = ~n117 & ~n126;
  assign n135 = ~n133 & ~n134;
  assign n136 = n132 & ~n135;
  assign n137 = ~n132 & n135;
  assign n138 = ~n136 & ~n137;
  assign n139 = ~x10 & x23;
  assign n140 = x10 & ~x23;
  assign n141 = ~n139 & ~n140;
  assign n144 = ~n132 & n134;
  assign n142 = ~n132 & n133;
  assign n143 = x9 & x22;
  assign n145 = ~n142 & ~n143;
  assign n146 = ~n144 & n145;
  assign n147 = n141 & ~n146;
  assign n148 = ~n141 & n146;
  assign n149 = ~n147 & ~n148;
  assign n150 = ~x11 & x24;
  assign n151 = x11 & ~x24;
  assign n152 = ~n150 & ~n151;
  assign n156 = ~n141 & n144;
  assign n154 = ~n141 & n142;
  assign n153 = ~n141 & n143;
  assign n155 = x10 & x23;
  assign n157 = ~n153 & ~n155;
  assign n158 = ~n154 & n157;
  assign n159 = ~n156 & n158;
  assign n160 = n152 & ~n159;
  assign n161 = ~n152 & n159;
  assign n162 = ~n160 & ~n161;
  assign n163 = ~x12 & x25;
  assign n164 = x12 & ~x25;
  assign n165 = ~n163 & ~n164;
  assign n170 = ~n152 & n156;
  assign n168 = ~n152 & n154;
  assign n167 = ~n152 & n153;
  assign n166 = ~n152 & n155;
  assign n169 = x11 & x24;
  assign n171 = ~n166 & ~n169;
  assign n172 = ~n167 & n171;
  assign n173 = ~n168 & n172;
  assign n174 = ~n170 & n173;
  assign n175 = n165 & ~n174;
  assign n176 = ~n165 & n174;
  assign n177 = ~n175 & ~n176;
  assign n178 = x12 & x25;
  assign n179 = ~n165 & ~n174;
  assign n180 = ~n178 & ~n179;
  assign y0 = ~n33;
  assign y1 = ~n42;
  assign y2 = ~n53;
  assign y3 = ~n66;
  assign y4 = ~n81;
  assign y5 = ~n90;
  assign y6 = ~n101;
  assign y7 = ~n114;
  assign y8 = ~n129;
  assign y9 = ~n138;
  assign y10 = ~n149;
  assign y11 = ~n162;
  assign y12 = ~n177;
  assign y13 = ~n180;
endmodule