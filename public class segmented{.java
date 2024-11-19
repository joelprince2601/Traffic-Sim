public class segmented{
    static void SegSieve(int l,int h)
    {
        boolean prime[] = new boolean[h + 1];
        for (int p=2;p*p<=h;p++) prime{
            int sm=(1/p)*p;
            if (sm<l) sm=p*(l/p+1);
            for (int i=sm;i<=h;i+=p) prime[i-l]=true;
        }
        for (int i=1;i<=h;i++){
            if(!prime[i]) System.out.print(i + " ");
            if(i%10==0) System.out.println();
            else if(i==h) System.out.println();

        }
    
    
}