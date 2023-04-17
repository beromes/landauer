module CRippleCarryAdder_14_14_14_InOutT(pA13,pA12,pA11,pA10,pA9,pA8,pA7,pA6,pA5,pA4,pA3,pA2,pA1,pA0, pB13,pB12,pB11,pB10,pB9,pB8,pB7,pB6,pB5,pB4,pB3,pB2,pB1,pB0, r13,r12,r11,r10,r9,r8,r7,r6,r5,r4,r3,r2,r1,r0, cIn, cOut);

input pA13,pA12,pA11,pA10,pA9,pA8,pA7,pA6,pA5,pA4,pA3,pA2,pA1,pA0, pB13,pB12,pB11,pB10,pB9,pB8,pB7,pB6,pB5,pB4,pB3,pB2,pB1,pB0, cIn;
output r13,r12,r11,r10,r9,r8,r7,r6,r5,r4,r3,r2,r1,r0, cOut;
wire w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34, w35, w36, w37, w38, w39, w40, w41, w42, w43, w44, w45, w46, w47, w48, w49, w50, w51, w52, w53, w54, w55, w56, w57, w58, w59, w60, w61, w62, w63, w64, w65, w66, w67, w68, w69, w70, w71, w72, w73, w74, w75, w76, w77, w78, w79, w80, w81, w82, w83, w84, w85, w86, w87, w88, w89, w90, w91, w92, w93, w94, w95, w96, w97, w98, w99, w100;

assign w1 = pA13;
assign w2 = pA12;
assign w3 = pA11;
assign w4 = pA10;
assign w5 = pA9;
assign w6 = pA8;
assign w7 = pA7;
assign w8 = pA6;
assign w9 = pA5;
assign w10 = pA4;
assign w11 = pA3;
assign w12 = pA2;
assign w13 = pA1;
assign w14 = pA0;
assign w15 = pB13;
assign w16 = pB12;
assign w17 = pB11;
assign w18 = pB10;
assign w19 = pB9;
assign w20 = pB8;
assign w21 = pB7;
assign w22 = pB6;
assign w23 = pB5;
assign w24 = pB4;
assign w25 = pB3;
assign w26 = pB2;
assign w27 = pB1;
assign w28 = pB0;
assign r13 = w29;
assign r12 = w30;
assign r11 = w31;
assign r10 = w32;
assign r9 = w33;
assign r8 = w34;
assign r7 = w35;
assign r6 = w36;
assign r5 = w37;
assign r4 = w38;
assign r3 = w39;
assign r2 = w40;
assign r1 = w41;
assign r0 = w42;
assign w44 = cIn;
assign cOut = w43;

assign w47 = w1 ^ w15;
assign w29 = w47 ^ w44;
assign w48 = w44 & w47;
assign w49 = w15 & w1;
assign w46 = w48 | w49;
assign w51 = w2 ^ w16;
assign w30 = w51 ^ w46;
assign w52 = w46 & w51;
assign w53 = w16 & w2;
assign w50 = w52 | w53;
assign w55 = w3 ^ w17;
assign w31 = w55 ^ w50;
assign w56 = w50 & w55;
assign w57 = w17 & w3;
assign w54 = w56 | w57;
assign w59 = w4 ^ w18;
assign w32 = w59 ^ w54;
assign w60 = w54 & w59;
assign w61 = w18 & w4;
assign w58 = w60 | w61;
assign w63 = w5 ^ w19;
assign w33 = w63 ^ w58;
assign w64 = w58 & w63;
assign w65 = w19 & w5;
assign w62 = w64 | w65;
assign w67 = w6 ^ w20;
assign w34 = w67 ^ w62;
assign w68 = w62 & w67;
assign w69 = w20 & w6;
assign w66 = w68 | w69;
assign w71 = w7 ^ w21;
assign w35 = w71 ^ w66;
assign w72 = w66 & w71;
assign w73 = w21 & w7;
assign w70 = w72 | w73;
assign w75 = w8 ^ w22;
assign w36 = w75 ^ w70;
assign w76 = w70 & w75;
assign w77 = w22 & w8;
assign w74 = w76 | w77;
assign w79 = w9 ^ w23;
assign w37 = w79 ^ w74;
assign w80 = w74 & w79;
assign w81 = w23 & w9;
assign w78 = w80 | w81;
assign w83 = w10 ^ w24;
assign w38 = w83 ^ w78;
assign w84 = w78 & w83;
assign w85 = w24 & w10;
assign w82 = w84 | w85;
assign w87 = w11 ^ w25;
assign w39 = w87 ^ w82;
assign w88 = w82 & w87;
assign w89 = w25 & w11;
assign w86 = w88 | w89;
assign w91 = w12 ^ w26;
assign w40 = w91 ^ w86;
assign w92 = w86 & w91;
assign w93 = w26 & w12;
assign w90 = w92 | w93;
assign w95 = w13 ^ w27;
assign w41 = w95 ^ w90;
assign w96 = w90 & w95;
assign w97 = w27 & w13;
assign w94 = w96 | w97;
assign w98 = w14 ^ w28;
assign w42 = w98 ^ w94;
assign w99 = w94 & w98;
assign w100 = w28 & w14;
assign w43 = w99 | w100;

endmodule
